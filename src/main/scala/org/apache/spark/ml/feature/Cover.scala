/*
 * Copyright 2018 Shingo OKAWA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark.ml.feature

import scala.collection.mutable.ArrayBuilder
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkException
import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.{Model, Estimator}
import org.apache.spark.ml.linalg.{Vector, Vectors, VectorUDT}
import org.apache.spark.ml.param.{
  IntParam,
  DoubleParam,
  BooleanParam,
  ParamValidators,
  Param,
  Params,
  ParamMap
}
import org.apache.spark.ml.param.shared.{HasInputCols, HasOutputCol}
import org.apache.spark.ml.util.{
  Identifiable,
  SchemaUtils,
  MLWritable,
  MLReadable,
  MLWriter,
  MLReader,
  DefaultParamsWriter,
  DefaultParamsReader,
  DefaultParamsWritable,
  DefaultParamsReadable
}
import org.apache.spark.ml.util.interval.{Cube, OpenCover, ClosedCover}
import org.apache.spark.mllib.linalg.{
  Vector => OldVector,
  Vectors => OldVectors
}
import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions.{
  udf,
  array,
  struct,
  col,
  explode,
  monotonically_increasing_id
}
import org.apache.spark.sql.types.{
  NumericType,
  DoubleType,
  StructType,
  ArrayType,
  LongType,
  IntegerType
}

/**
  * Params for [[Cover]] and [[CoverModel]].
  *
  * @author Shingo OKAWA
  */
private[feature] trait CoverParams
    extends Params
    with HasInputCols
    with HasOutputCol {

  import CoverModel._

  /**
    * Number of interval splits. Should be greater than 0.
    * Default: 10
    * @group param
    */
  final val numSplits = new IntParam(this,
                                     "numSplits",
                                     "number of interval splits (> 0)",
                                     ParamValidators.gt(0))

  /** @group getParam */
  final def getNumSplits: Int = $(numSplits)

  /**
    * Overlap ratio of each interval. Should be greater than 0.0.
    * Default: 0.1
    * @group param
    */
  final val overlapRatio = new DoubleParam(
    this,
    "overlapRatio",
    "overlap ration of each interval (> 0)",
    ParamValidators.gt(0))

  /** @group getParam */
  final def getOverlapRatio: Double = $(overlapRatio)

  /**
    * If this value is set to be true, the resulting `DataFrame` will be exploded with the specified output column.
    * Default: false
    * @group param
    */
  final val exploding = new BooleanParam(
    this,
    "exploding",
    "If True, the resulting DataFrame will be exploded with the specified output column.")

  /** @group getParam */
  final def getExploding: Boolean = $(exploding)

  /**
    * Param for data identifier column name.
    * Default: "id"
    * @group param
    */
  final val idCol: Param[String] =
    new Param[String](this, "idCol", "data identifier column name")

  /** @group getParam */
  final def getIdCol: String = $(idCol)

  /**
    * Validates and transforms the input schema.
    */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    val fields = schema($(inputCols).toSet)
    fields.foreach { field =>
      val (dtype, name) = (field.dataType, field.name)
      require(
        (dtype.isInstanceOf[NumericType] || dtype.isInstanceOf[VectorUDT]),
        s"Cover requires columns to be of NumericType. Column $name was $dtype")
    }
    if ($(exploding)) {
      val newSchema =
        SchemaUtils.appendColumn(schema, $(outputCol), IntegerType)
      SchemaUtils.appendColumn(newSchema, $(idCol), LongType)
    } else {
      SchemaUtils.appendColumn(schema,
                               $(outputCol),
                               ArrayType(IntegerType, false))
    }
  }
}

/**
  * An `Assembler` is responsible to assemble multiple column values into a single vector.
  */
private[feature] trait Assembler {

  /** Assembles given column values into single vector. */
  def assemble(columns: Any*): Vector = {
    val (indices, values) = (ArrayBuilder.make[Int], ArrayBuilder.make[Double])
    var current = 0
    columns.foreach {
      case value: Double =>
        if (value != 0.0) {
          indices += current
          values += value
        }
        current += 1
      case vector: Vector =>
        vector.foreachActive {
          case (i, value) =>
            if (value != 0.0) {
              indices += current + i
              values += value
            }
        }
        current += vector.size
      case null =>
        throw new SparkException("values to assemble cannot be null")
      case other =>
        throw new SparkException(
          s"$other of type ${other.getClass.getName} is not supported")
    }
    Vectors.sparse(current, indices.result(), values.result()).compressed
  }

  /** A Spark/SQL user defined function to assemble a given `Row`. */
  val vectorizeUDF = udf { row: Row =>
    assemble(row.toSeq: _*)
  }
}

/**
  * Cover generates overlapping intervals, i.e., hypercubes and breaks down the given point cloud dataframe into
  * overlapping bins with the generated intervals by assigning an identifier array into the output column. This is
  * done by projecting a set of numerical columns into a single feature vector and caluculating each columns'
  * minimum/maximum values using column summary statistics.
  *
  * The [[Cover]] estimator operates on multiple columns. Each column must contain numeric features.
  *
  * {{{
  *   val df = Seq(
  *    ( 36, 51.472459345502074,  true, "GHIJ"),
  *    ( 36, 51.472459345502074,  true, "GHIJ"),
  *    (-97, -75.82637174773971, false,  "DEF"),
  *    (-97, -75.82637174773971, false,  "DEF"),
  *    (-97, -75.82637174773971, false,  "DEF"),
  *    (  1, -16.66589911618783,  true, "GHIJ"),
  *    (  1, -16.66589911618783,  true, "GHIJ"),
  *    (-99, 23.506155529566698,  true, "GHIJ"),
  *    (-99, 23.506155529566698,  true, "GHIJ"),
  *    (-99, 23.506155529566698,  true, "GHIJ")
  *   ).toDF("integer", "double", "boolean", "string")
  *
  *   val cover = new Cover()
  *    .setInputCols("double", "integer")
  *    .setOutputCol("cover_ids")
  *    .setOverlapRatio(0.5)
  *    .setNumSplits(3)
  *
  *   val model = cover.fit(df)
  *   model.transform(df).show()
  *
  *   +-------+------------------+-------+------+---------+
  *   |integer|            double|boolean|string|cover_ids|
  *   +-------+------------------+-------+------+---------+
  *   |     36|51.472459345502074|   true|  GHIJ|      [8]|
  *   |     36|51.472459345502074|   true|  GHIJ|      [8]|
  *   |    -97|-75.82637174773971|  false|   DEF|      [0]|
  *   |    -97|-75.82637174773971|  false|   DEF|      [0]|
  *   |    -97|-75.82637174773971|  false|   DEF|      [0]|
  *   |      1|-16.66589911618783|   true|  GHIJ|   [4, 5]|
  *   |      1|-16.66589911618783|   true|  GHIJ|   [4, 5]|
  *   |    -99|23.506155529566698|   true|  GHIJ|      [6]|
  *   |    -99|23.506155529566698|   true|  GHIJ|      [6]|
  *   |    -99|23.506155529566698|   true|  GHIJ|      [6]|
  *   +-------+------------------+-------+------+---------+
  *
  *   model.cubes.foreach(println)
  *
  *   Cube(id=0, (-86.43460767217653, -22.785192125555636), (-110.25, -42.75))
  *   Cube(id=1, (-86.43460767217653, -22.785192125555636), (-65.25, 2.25))
  *   Cube(id=2, (-86.43460767217653, -22.785192125555636), (-20.25, 47.25))
  *   Cube(id=3, (-44.00166397442926, 19.647751572191627), (-110.25, -42.75))
  *   Cube(id=4, (-44.00166397442926, 19.647751572191627), (-65.25, 2.25))
  *   Cube(id=5, (-44.00166397442926, 19.647751572191627), (-20.25, 47.25))
  *   Cube(id=6, (-1.5687202766820043, 62.080695269938886), (-110.25, -42.75))
  *   Cube(id=7, (-1.5687202766820043, 62.080695269938886), (-65.25, 2.25))
  *   Cube(id=8, (-1.5687202766820043, 62.080695269938886), (-20.25, 47.25))
  * }}}
  *
  * @author Shingo OKAWA
  */
@Experimental
class Cover(override val uid: String)
    extends Estimator[CoverModel]
    with CoverParams
    with DefaultParamsWritable {

  import Cover._

  def this() = this(Identifiable.randomUID("cover"))

  setDefault(numSplits -> 10,
             overlapRatio -> 0.1,
             exploding -> false,
             idCol -> "id")

  /** @group setParam */
  def setInputCols(values: String*): this.type = setInputCols(values.toArray)

  /** @group setParam */
  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  /** @gr oup setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group setParam */
  def setNumSplits(value: Int): this.type = set(numSplits, value)

  /** @group setParam */
  def setOverlapRatio(value: Double): this.type = set(overlapRatio, value)

  /** @group setParam */
  def setExploding(value: Boolean): this.type = set(exploding, value)

  /** @group setParam */
  def setIdCol(value: String): this.type = set(idCol, value)

  /**
    * Fits a model to the input data.
    */
  override def fit(dataset: Dataset[_]): CoverModel = {
    transformSchema(dataset.schema, logging = true)
    val args = $(inputCols).map { col =>
      dataset.schema(col).dataType match {
        case DoubleType => dataset(col)
        case _: VectorUDT => dataset(col)
        case _: NumericType =>
          dataset(col).cast(DoubleType).as(s"${col}_double_$uid")
      }
    }
    val input: RDD[OldVector] =
      dataset.select(vectorizeUDF(struct(args: _*))).rdd.map {
        case Row(v: Vector) => OldVectors.fromML(v)
      }
    val summary = Statistics.colStats(input)
    copyValues(new CoverModel(uid, summary.min, summary.max).setParent(this))
  }

  /**
    * Transforms the input dataset's schema.
    */
  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  /**
    * Copies [[Cover]] with extra parameters.
    */
  override def copy(extra: ParamMap): Cover = defaultCopy(extra)
}

object Cover extends DefaultParamsReadable[Cover] with Assembler {

  /** Reads an ML instance from the input path, a shortcut of read.load(path). */
  override def load(path: String): Cover = super.load(path)
}

/**
  * Model fitted by [[Cover]].
  *
  * @param min minimum values for each input columns during fitting
  * @param max maximum values for each input columns during fitting
  *
  * @author Shingo OKAWA
  */
@Experimental
class CoverModel private[ml] (override val uid: String,
                              val min: Vector,
                              val max: Vector)
    extends Model[CoverModel]
    with CoverParams
    with MLWritable {

  import CoverModel._

  /** @group setParam */
  def setInputCols(values: String*): this.type = setInputCols(values.toArray)

  /** @group setParam */
  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group setParam */
  def setNumSplits(value: Int): this.type = set(numSplits, value)

  /** @group setParam */
  def setOverlapRatio(value: Double): this.type = set(overlapRatio, value)

  /** @group setParam */
  def setExploding(value: Boolean): this.type = set(exploding, value)

  /** @group setParam */
  def setIdCol(value: String): this.type = set(idCol, value)

  /**
    * Instanciates an open cover of the specified min-max arrays.
    */
  def cubes: Seq[Cube] = {
    val (minArr, maxArr) = (min.asBreeze.toArray, max.asBreeze.toArray)
    require(minArr.length == maxArr.length)
    OpenCover($(numSplits), $(overlapRatio), minArr zip maxArr)
  }

  /**
    * Transforms the input dataset.
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val cubes = this.cubes
    val assignment = udf { xs: Vector =>
      val seenAt = ArrayBuilder.make[Int]
      for (u <- cubes) { if (u contains Seq(xs.toArray: _*)) seenAt += u.id }
      seenAt.result()
    }
    val args = $(inputCols).map { col =>
      dataset.schema(col).dataType match {
        case DoubleType => dataset(col)
        case _: VectorUDT => dataset(col)
        case _: NumericType =>
          dataset(col).cast(DoubleType).as(s"${col}_double_$uid")
      }
    }
    val result = dataset.withColumn(s"${$(outputCol)}_${uid}",
                                    assignment(vectorizeUDF(struct(args: _*))))
    if ($(exploding))
      result
        .withColumn($(idCol), monotonically_increasing_id())
        .withColumn($(outputCol), explode(col(s"${$(outputCol)}_${uid}")))
        .drop(s"${$(outputCol)}_${uid}")
    else result.withColumnRenamed(s"${$(outputCol)}_${uid}", $(outputCol))
  }

  /**
    * Transforms the input dataset's schema.
    */
  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  /**
    * Copies [[CoverModel]] with extra parameters.
    */
  override def copy(extra: ParamMap): CoverModel = {
    val copied = new CoverModel(uid, min, max)
    copyValues(copied, extra).setParent(parent)
  }

  /**
    * Returns an [[MLWriter]] instance for this ML instance.
    */
  override def write: MLWriter = new CoverModelWriter(this)
}

object CoverModel extends MLReadable[CoverModel] with Assembler {

  /** Holds path to the directory where data dumped. */
  private val PARENT_PATH: String = "data"

  /** Holds column name for minimum values. */
  private val COL_MIN: String = "min"

  /** Holds column name for maximum values. */
  private val COL_MAX: String = "max"

  /** [[MLWriter]] for [[CoverModel]]. */
  private[CoverModel] class CoverModelWriter(instance: CoverModel)
      extends MLWriter {

    /** Serializable adhoc class for writing data. */
    private case class Data(min: Vector, max: Vector)

    /** `save()` handles overwriting and then calls this method.*/
    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      sparkSession
        .createDataFrame(Seq(new Data(instance.min, instance.max)))
        .repartition(1)
        .write
        .parquet(new Path(path, PARENT_PATH).toString)
    }
  }

  /** [[MLReader]] for [[CoverModel]]. */
  private class CoverModelReader extends MLReader[CoverModel] {

    /** Loads the ML component from the input path. */
    override def load(path: String): CoverModel = {
      val metadata =
        DefaultParamsReader.loadMetadata(path, sc, classOf[CoverModel].getName)
      val Row(min: Vector, max: Vector) =
        MLUtils
          .convertVectorColumnsToML(
            sparkSession.read.parquet(new Path(path, PARENT_PATH).toString),
            COL_MIN,
            COL_MAX)
          .select(COL_MIN, COL_MAX)
          .head()
      val model = new CoverModel(metadata.uid, min, max)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

  /** Returns an [[MLReader]] instance for this class. */
  override def read: MLReader[CoverModel] = new CoverModelReader

  /** Reads an ML instance from the input path, a shortcut of read.load(path). */
  override def load(path: String): CoverModel = super.load(path)
}
