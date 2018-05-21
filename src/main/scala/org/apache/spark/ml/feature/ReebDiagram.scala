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

import scala.reflect.ClassTag
import scala.collection.mutable.ArrayBuffer
import scala.util.hashing.byteswap64
import breeze.linalg.DenseVector
import breeze.stats.{MeanAndVariance, mean, meanAndVariance}
import org.apache.spark.HashPartitioner
import org.apache.spark.annotation.Experimental
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.graphx.{Graph, EdgeTriplet, VertexId, Edge}
import org.apache.spark.ml.{Model, Estimator}
import org.apache.spark.ml.linalg.{VectorUDT, Distance, Vector}
import org.apache.spark.ml.param.{
  Param,
  Params,
  ParamMap,
  IntParam,
  LongParam,
  DoubleParam,
  IntArrayParam,
  ParamValidators
}
import org.apache.spark.ml.param.shared.{HasOutputCol, HasFeaturesCol, HasSeed}
import org.apache.spark.ml.util.{SchemaUtils, Identifiable}
import org.apache.spark.ml.util.knn.{
  TopTrees,
  Tree,
  Branch,
  VPTree,
  VectorWithId,
  VectorEntry,
  TopTreesPartitioner
}
import org.apache.spark.mllib.linalg.{
  Vector => OldVector,
  SparseVector => OldSparseVector,
  Vectors => OldVectors
}
import org.apache.spark.mllib.linalg.distributed.{
  IndexedRow,
  IndexedRowMatrix,
  CoordinateMatrix,
  MatrixEntry
}
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrixFunctions._
import org.apache.spark.mllib.rdd.MLPairRDDFunctions._
import org.apache.spark.rdd.{RDD, ShuffledRDD}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.types.{
  ArrayType,
  IntegerType,
  LongType,
  StructType
}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.random.XORShiftRandom

/**
  * Params for [[ReebDiagram]] and [[ReebDiagramModel]].
  *
  * @author Shingo OKAWA
  */
private[feature] trait ReebDiagramParams
    extends Params
    with HasFeaturesCol
    with HasOutputCol {

  /**
    * Param for input cover column name.
    * @group param
    */
  final val coverCol: Param[String] =
    new Param[String](this, "coverCol", "input cover column name")

  /** @group getParam */
  final def getCoverCol: String = $(coverCol)

  /**
    * Param for data identifier column name.
    * @group param
    */
  final val idCol: Param[String] =
    new Param[String](this, "idCol", "data identifier column name")

  /** @group getParam */
  final def getIdCol: String = $(idCol)

  /**
    * Param for random seed.
    * Default: 0L
    * @group param
    */
  final val seed =
    new LongParam(this, "seed", "seed for generating random numbers")

  /** @group getParam */
  final def getSeed: Long = $(seed)

  /**
    * Number of data points to sample for top-level tree (> 0).
    * Default: 1000
    * @group param
    */
  final val topTreeSize = new IntParam(
    this,
    "topTreeSize",
    "number of data points to sample for top-level tree",
    ParamValidators.gt(0))

  /** @group getParam */
  final def getTopTreeSize: Int = $(topTreeSize)

  /**
    * Number of data points at which to switch to brute-force for top-level tree (> 0).
    * Default: 5
    * @group param
    */
  final val topTreeLeafSize = new IntParam(
    this,
    "topTreeLeafSize",
    "number of data points at which to switch to brute-force for top-level tree",
    ParamValidators.gt(0))

  /** @group getParam */
  final def getTopTreeLeafSize: Int = $(topTreeLeafSize)

  /**
    * Number of data points at which to switch to brute-force for sub-tree (> 0).
    * Default: 20
    * @group param
    */
  final val subTreeLeafSize = new IntParam(
    this,
    "subTreeLeafSize",
    "number of data points at which to switch to brute-force for sub-tree",
    ParamValidators.gt(0))

  /** @group getParam */
  final def getSubTreeLeafSize: Int = $(subTreeLeafSize)

  /**
    * Number of neighbors to find. Should be greater than 0.
    * Default: 5
    * @group param
    */
  final val k = new IntParam(this,
                             "k",
                             "number of neighbors to find",
                             ParamValidators.gt(0))

  /** @group getParam */
  final def getK: Int = $(k)

  /**
    * Weight coefficients to the number of neighbors which determines the edge is active or not.
    * Default: 0.4
    * @group param
    */
  final val linkThresholdRatio = new DoubleParam(
    this,
    "linkthresholdRatio",
    "weight coefficients to the number of neighbors which determines the edge is active or not",
    ParamValidators.inRange(0.0, 1.0, false, true)
  )

  /** @group getParam */
  final def getLinkThresholdRatio = $(linkThresholdRatio)

  /**
    * Weight coefficient to the number of neighbors which determines the data point is core point or not.
    * Default: 0.7
    * @group param
    */
  final val coreThresholdRatio = new DoubleParam(
    this,
    "coreThresholdRatio",
    "Weight coefficient to the number of neighbors which determines the data point is core point or not",
    ParamValidators.inRange(0.0, 1.0, false, true)
  )

  /** @group getParam */
  final def getCoreThresholdRatio = $(coreThresholdRatio)

  /**
    * Specifiees the measure of underlying data points.
    * Default: "euclidean"
    * @group param
    */
  final val measure: Param[String] =
    new Param(this, "measure", "measure of data points")

  /** @group getParam */
  final def getMeasure: String = $(measure)

  /**
    * Number of candidates of tree indexing partitioning data data points.
    * Default: 10
    * @group param
    */
  final val numVPCadidates = new IntParam(
    this,
    "numVPCandidates",
    "number of candidates of tree index partitioning data points",
    ParamValidators.gt(0))

  /** @group getParam */
  final def getNumVPCandidates: Int = $(numVPCadidates)

  /**
    * Number of sample data points to estimate a vantage point.
    * Default: 100
    * @group param
    */
  final val numVPSamples = new IntParam(
    this,
    "numVPSamples",
    "number of sample data points to estimate a vantage point",
    ParamValidators.gt(0)
  )

  /** @group param */
  final def getNumVPSamples: Int = $(numVPSamples)

  /**
    * Size of buffer used to spilling metric tree search. When buffer size is -1.0, it will trigger automatic
    * effective nearest neighbor distance estimation.
    * Default: -1.0
    * @group param
    */
  final val bufferSize = new DoubleParam(
    this,
    "bufferSize",
    "size of buffer used to spilling metric tree search",
    ParamValidators.gtEq(-1.0)
  )

  /** @group param */
  final def getBufferSize: Double = $(bufferSize)

  /**
    * Number of sample data points to estimate buffer size.
    * Default: 100 to 1000 by 100
    * @group param
    */
  final val bufferSizeSamples = new IntArrayParam(
    this,
    "bufferSizeSamples",
    "number of sample data points to estimate buffer size", {
      array: Array[Int] =>
        array.length > 1 && array.forall(_ > 0)
    }
  )

  /** @group param */
  final def getBufferSizeSamples: Array[Int] = $(bufferSizeSamples)

  // TODO: would be better to refactor this method to another trait.
  // TODO: cover ids can be caluculated without any redundancy with CoverModel.
  /** Returns the set of cover ids. */
  final def getCoverIds(dataset: Dataset[_]): Seq[Int] =
    dataset
      .select($(coverCol))
      .distinct
      .rdd
      .map(_.getInt(0))
      .collect()

  // TODO: would be better to refactor this method to another trait.
  /** Transform data points with duplications to vector with identifier. */
  final def getVectorEntries(
      dataset: Dataset[_]): RDD[(Int, Long, VectorEntry)] =
    dataset
      .select($(coverCol), $(idCol), $(featuresCol))
      .rdd
      .zipWithIndex
      .map {
        case (row, i) =>
          (row.getInt(0), row.getLong(1), VectorEntry(i, row.getAs[Vector](2)))
      }

  // TODO: would be better to refactor this method to another trait.
  /** Transforms vector entries with duplications to data points without duplication. */
  final def getDataPoints(
      vectorEntries: RDD[(Int, Long, VectorEntry)]): RDD[VectorEntry] =
    vectorEntries
      .map { case (_, id, entry) => (id, entry) }
      .reduceByKey { case (lhs, rhs) => lhs }
      .map { case (_, entry) => entry }

  /**
    * Validates and transforms the input schema.
    */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    SchemaUtils.checkColumnType(schema, $(coverCol), IntegerType)
    SchemaUtils.checkColumnType(schema, $(idCol), LongType)
    SchemaUtils.appendColumn(schema, $(outputCol), LongType)
  }
}

/**
  * ReebDiagram assigns cluster ids for each data entries by clustering data points within each covers, i.e., the
  * clustering will be done in each separated "dataset"s all at once. A `ReebDiagramModel` faciliates reeb diagram
  * construction by SNN-DBSCAN clustering for each covers.
  *
  * {{{
  *   val df = Seq(
  *     ( 73, -13.214398224760757,   true, KLMNO, Vectors.dense(-13.214398224760,  73), 274877906945, 48),
  *     (-71,  90.376652247800370,  false,  GHIJ, Vectors.dense(90.3766522478003, -71),            0, 91),
  *     (-49, -83.648455478759250,  false, KLMNO, Vectors.dense(-83.648455478759, -49), 120259084289,  2),
  *     ( 98,  69.390365339787650,   true,  GHIJ, Vectors.dense(69.3903653397876,  98), 429496729601, 89),
  *     (-84,  67.204683513074770,  false, KLMNO, Vectors.dense(67.2046835130747, -84), 438086664192, 80),
  *     (-71,  90.376652247800370,  false,  GHIJ, Vectors.dense(90.3766522478003, -71),            1, 91),
  *     ( 38, -99.043573525374650,  false,   DEF, Vectors.dense(-99.043573525374,  38), 283467841536,  6),
  *     // more data entries follows.
  *   ).toDF("integer", "double", "boolean", "string", "feature", "id", "cover_id")
  *
  *  val reeb = new ReebDiagram()
  *    .setK(15)
  *    .setIdCol("id")
  *    .setCoverCol("cover_id")
  *    .setFeaturesCol("feature")
  *    .setOutputCol("cluster_id")
  *
  *  reeb.fit(df).transform(df).show()
  *
  *  +-------+-------------------+-------+------+--------------------+------------+--------+----------+
  *  |integer|             double|boolean|string|              vector|          id|cover_id|cluster_id|
  *  +-------+-------------------+-------+------+--------------------+------------+--------+----------+
  *  |     73|-13.214398224760757|   true| KLMNO|[-13.214398224760...|274877906945|      48|        84|
  *  |    -71|  90.37665224780037|  false|  GHIJ|[90.3766522478003...|           0|      91|         0|
  *  |    -49| -83.64845547875925|  false| KLMNO|[-83.648455478759...|120259084289|       2|        42|
  *  |     98|  69.39036533978765|   true|  GHIJ|[69.3903653397876...|429496729601|      89|       126|
  *  |    -84|  67.20468351307477|  false| KLMNO|[67.2046835130747...|438086664192|      80|       127|
  *  |    -71|  90.37665224780037|  false|  GHIJ|[90.3766522478003...|           1|      91|         1|
  *  |     38| -99.04357352537465|  false|   DEF|[-99.043573525374...|283467841536|       6|        85|
  *  +-------+-------------------+-------+------+--------------------+------------+--------+----------+
  * }}}
  *
  * @author Shingo OKAWA
  */
@Experimental
class ReebDiagram(override val uid: String)
    extends Estimator[ReebDiagramModel]
    with ReebDiagramParams {

  import ReebDiagram._

  def this() = this(Identifiable.randomUID("reeb_diagram"))

  setDefault(
    seed -> 0L,
    topTreeSize -> 1000,
    topTreeLeafSize -> 10,
    subTreeLeafSize -> 30,
    k -> 5,
    linkThresholdRatio -> 0.4,
    coreThresholdRatio -> 0.7,
    measure -> "euclidean",
    numVPCadidates -> 10,
    numVPSamples -> 100,
    bufferSize -> -1.0,
    bufferSizeSamples -> (100 to 1000 by 100).toArray
  )

  /** @group setParam */
  def setCoverCol(value: String): this.type = set(coverCol, value)

  /** @group setParam */
  def setIdCol(value: String): this.type = set(idCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group setParam */
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** @group setParam */
  def setSeed(value: Long): this.type = set(seed, value)

  /** @group setParam */
  def setTopTreeSize(value: Int): this.type = set(topTreeSize, value)

  /** @group setParam */
  def setTopTreeLeafSize(value: Int): this.type = set(topTreeLeafSize, value)

  /** @group setParam */
  def setSubTreeLeafSize(value: Int): this.type = set(subTreeLeafSize, value)

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

  /** @group setParam */
  def setLinkThresholdRatio(value: Double): this.type =
    set(linkThresholdRatio, value)

  /** @group setParam */
  def setCoreThresholdRatio(value: Double): this.type =
    set(coreThresholdRatio, value)

  /** @group setParam */
  def setMeasure(value: String): this.type = set(measure, value)

  /** @group setParam */
  def setNumVPCandidates(value: Int): this.type = set(numVPCadidates, value)

  /** @group setParam */
  def setNumVPSamples(value: Int): this.type = set(numVPSamples, value)

  /** @group setParam */
  def setBufferSize(value: Double): this.type = set(bufferSize, value)

  /** @group setParam */
  def setBufferSizeSamples(value: Array[Int]): this.type =
    set(bufferSizeSamples, value)

  /**
    * Fits a model to the input data.
    */
  override def fit(dataset: Dataset[_]): ReebDiagramModel = {
    val rand = new XORShiftRandom($(seed))
    // Collect predefined cover ids.
    val coverIds = getCoverIds(dataset)
    // Transform data points to vector with identifier.
    val vectorEntries = getVectorEntries(dataset)
    // Build top-trees to repartition data points.
    val topTrees = TopTrees(coverIds.map { coverId =>
      val cover = vectorEntries.filter {
        case (cid, _, entry) => cid == coverId
      }
      val sampled = cover
        .takeSample(withReplacement = false, $(topTreeSize), rand.nextLong())
      (coverId,
       VPTree(sampled.map { case (_, _, v) => v },
              Distance($(measure)),
              $(numVPCadidates),
              $(numVPSamples),
              $(topTreeLeafSize),
              rand.nextLong()))
    }.toSeq)
    // Repartition data points.
    val partitioner = new TopTreesPartitioner(topTrees)
    val repartitioned =
      new ShuffledRDD[(Int, VectorWithId), Null, Null](vectorEntries.map {
        case (cid, _, v) => ((cid, v), null)
      }, partitioner).keys
    // Build sub-trees to index data points.
    val subTrees = repartitioned
      .mapPartitionsWithIndex {
        case (partitionId, iter) =>
          val rand = new XORShiftRandom(byteswap64($(seed) ^ partitionId))
          Iterator(
            VPTree(
              iter.map { case (_, v) => v }.toIndexedSeq,
              Distance($(measure)),
              $(numVPCadidates),
              $(numVPSamples),
              $(subTreeLeafSize),
              rand.nextLong()
            ))
      }
      .persist(StorageLevel.MEMORY_AND_DISK)
    // Estimate averaged minimum distance
    val estimatedBufferSize = if ($(bufferSize) < 0) {
      val dataPoints = getDataPoints(vectorEntries)
      EstimatedBufferSize(dataPoints,
                          $(k),
                          Distance($(measure)),
                          $(bufferSizeSamples),
                          rand.nextLong())
    } else {
      math.max(0, $(bufferSize))
    }
    // Construct Reeb diagram model.
    val model =
      new ReebDiagramModel(uid, subTrees.context.broadcast(topTrees), subTrees)
        .setParent(this)
    copyValues(model).setBufferSize(estimatedBufferSize)
  }

  /**
    * Transforms the input dataset's schema.
    */
  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  /**
    * Copies [[ReebDiagram]] with extra parameters.
    */
  override def copy(extra: ParamMap): ReebDiagram = defaultCopy(extra)
}

object ReebDiagram {

  /**
    * Estimates a suitable distance buffer size based on a dataset. A suitable buffer size is the minimum size such
    * that nearest neighbors can be accurately found even at boundary of splitting circle arround vantage points.
    * Therefor assuming points are uniformly distributed in high dimensional space, it should be approximately the
    * average distance between points.
    *
    * Specifically the number of points within a certain radius of a given point is proportionally to the density of
    * points raised to the effective number of dimensions, of which manifold data points exist on:
    *
    *   R_s = \frac{c}{N_s ** 1/d}
    *
    * where R_s is the radius, N_s is the number of points, d is effective number of dimension, and c is a constant.
    * To estimate R_s_all for entire dataset, we can take samples of the dataset of different size N_s to compute
    * R_s. We can estimate c and d using linear regression. Lastly we can calculate R_s_all using total number of
    * observation in dataset.
    *
    * @see [[https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/32616.pdf
    *        Clustering Billions of Images with Large Scale Nearest Neighbor Search]]
    * @see [[http://kenclarkson.org/nn_survey/p.pdf
    *        Nearest-Neighbor Searching and Metric Space Dimensions]]
    * @param data the data points to be estimated
    * @param quantity the number of k nearest neighbors
    * @param measure the underlying distance measure to estimate
    * @param sampleSizes sample sizes to estimate average distance
    * @param seed random seed
    * @return estimated average distance of data points
    */
  private[feature] def EstimatedBufferSize(data: RDD[VectorEntry],
                                           quantity: Int,
                                           measure: Distance,
                                           sampleSizes: Array[Int],
                                           seed: Long): Double = {
    val cardinality = data.count()
    // Take samples of data points for estimation
    val samples = data.mapPartitionsWithIndex {
      case (partitionId, iter) =>
        val rand = new XORShiftRandom(byteswap64(seed ^ partitionId))
        iter.flatMap { vectorWithId =>
          sampleSizes.zipWithIndex
            .filter {
              case (size, _) => rand.nextDouble() * cardinality < size
            }
            .map { case (size, index) => (index, vectorWithId) }
        }
    }
    // Create estimators
    // TODO: Avoid using `groupByKey`
    val estimators = samples
      .groupByKey()
      .map {
        case (index, vectorWithIds) =>
          (vectorWithIds.size, MeanRadius(vectorWithIds, quantity, measure))
      }
      .collect()
      .distinct
    // x-y plot for linear regression
    val x = DenseVector(estimators.map { case (ns, _) => math.log(ns) })
    val y = DenseVector(estimators.map { case (_, rs) => math.log(rs) })
    // Estimate log(R_s) = alpha + beta * log(N_s)
    val xMeanVariance: MeanAndVariance = meanAndVariance(x)
    val xMean = xMeanVariance.mean
    val yMeanVariance: MeanAndVariance = meanAndVariance(y)
    val yMean = yMeanVariance.mean
    val correlation =
      (mean(x *:* y) - xMean * yMean) / math.sqrt(
        (mean(x *:* x) - xMean * xMean) * (mean(y *:* y) - yMean * yMean))
    val beta = correlation * yMeanVariance.stdDev / xMeanVariance.stdDev
    val alpha = yMean - beta * xMean
    val rs = math.exp(alpha + beta * math.log(cardinality))
    // Return result TODO: fix the case where beta is positive.
    if (beta > 0 || beta.isNaN || rs.isNaN) {
      math.exp(breeze.linalg.max(y))
    } else {
      rs / math.sqrt(-1.0 / beta)
    }
  }

  /** Computes the average distance of nearest neighbors within points using brute-force. */
  private def MeanRadius(data: Iterable[VectorWithId],
                         quantity: Int,
                         measure: Distance): Double = {
    val distances = data
      .map { pivot =>
        data
          .map(vectorWithId => measure(pivot.vector, vectorWithId.vector))
          .filter(_ > 0)
          .toList
          .sortWith(_ < _)
          .take(quantity)
          .last
      }
    distances.sum / distances.size
  }
}

/**
  * Model fitted by [[ReebDiagram]]. A `ReebDiagramModel` is not designed to work with `Tree`s that have not been
  * cached.
  *
  * @param topTrees a top-level tree to index data points
  * @param subTrees indexed data points to query kNN search
  *
  * @author Shingo OKAWA
  */
@Experimental
class ReebDiagramModel private[ml] (override val uid: String,
                                    val topTrees: Broadcast[TopTrees],
                                    val subTrees: RDD[Tree])
    extends Model[ReebDiagramModel]
    with ReebDiagramParams {

  require(
    subTrees.getStorageLevel != StorageLevel.NONE,
    "ReebDiagramModel is not designed to work with Trees that have not been cached")

  import ReebDiagramModel._

  /** @group setParam */
  def setCoverCol(value: String): this.type = set(coverCol, value)

  /** @group setParam */
  def setIdCol(value: String): this.type = set(idCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group setFeaturesCol */
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** @group setParam */
  def setSeed(value: Long): this.type = set(seed, value)

  /** @group setParam */
  def setTopTreeSize(value: Int): this.type = set(topTreeSize, value)

  /** @group setParam */
  def setTopTreeLeafSize(value: Int): this.type = set(topTreeLeafSize, value)

  /** @group setParam */
  def setSubTreeLeafSize(value: Int): this.type = set(subTreeLeafSize, value)

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

  /** @group setParam */
  def setLinkThresholdRatio(value: Double): this.type =
    set(linkThresholdRatio, value)

  /** @group setParam */
  def setCoreThresholdRatio(value: Double): this.type =
    set(coreThresholdRatio, value)

  /** @group setParam */
  def setMeasure(value: String): this.type = set(measure, value)

  /** @group setParam */
  def setNumVPCandidates(value: Int): this.type = set(numVPCadidates, value)

  /** @group setParam */
  def setNumVPSamples(value: Int): this.type = set(numVPSamples, value)

  /** @group setParam */
  def setBufferSize(value: Double): this.type = set(bufferSize, value)

  /** @group setParam */
  def setBufferSizeSamples(value: Array[Int]): this.type =
    set(bufferSizeSamples, value)

  /**
    * Transforms the input datasets's schema.
    */
  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  /**
    * Transforms the input dataset.
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    // Query kNN for each data points.
    val kNNs: RDD[(Long, Array[(Long, Double)])] =
      KNearestNeighbors(getVectorEntries(dataset), topTrees, subTrees)
    // Construct Ertoes graph.
    val ertoezGraph =
      toErtoezGraph(kNNs, $(k), $(linkThresholdRatio), $(coreThresholdRatio))
    // Get the column number of cover ids to handle noise data.
    val indexOfCoverId = dataset.toDF.schema.fieldIndex($(coverCol))
    dataset.sqlContext.createDataFrame(
      dataset.toDF.rdd.zipWithIndex
        .map {
          case (row, i) =>
            (i, row)
        }
        .leftOuterJoin(ertoezGraph.connectedComponents.vertices)
        .map {
          case (i, (row, clusterId)) =>
            Row.fromSeq(
              row.toSeq :+ clusterId.getOrElse(
                -(row.getInt(indexOfCoverId).toLong + 1L)))
        },
      transformSchema(dataset.schema)
    )
  }

  /**
    * Collects kNN data identifiers for each vector entries.
    *
    * @param vectorEntries data points to be queried
    * @param topTrees top-level VP tree structure to index data points
    * @param subTrees previouslt indexed data points to query kNN for each data points
    * @return kNN for each vector entries
    */
  private def KNearestNeighbors(
      vectorEntries: RDD[(Int, Long, VectorEntry)],
      topTrees: Broadcast[TopTrees],
      subTrees: RDD[Tree]): RDD[(Long, Array[(Long, Double)])] = {
    // Repartition data points to query kNN.
    val repartitioned = vectorEntries
      .flatMap {
        case (cid, id, vectorWithId) =>
          val indices =
            search((cid, vectorWithId), topTrees.value, $(bufferSize))
              .map(i => (i, vectorWithId))
          assert(indices.nonEmpty,
                 s"indices must be non empty: $cid, $vectorWithId")
          indices
      }
      .partitionBy(new HashPartitioner(subTrees.partitions.length))
    // Query kNN candidates for each data points.
    val candidates = repartitioned.zipPartitions(subTrees) {
      case (iter, trees) =>
        val tree = trees.next()
        assert(!trees.hasNext,
               s"sub-level tree must be uniquely defined for each indices")
        iter.flatMap {
          case (_, vectorWithId) =>
            tree.query(vectorWithId, $(k)).map {
              case (neighbor, distance) =>
                (vectorWithId.id, (neighbor.id, distance))
            }
        }
    }
    // Merge results by point index together and keep track of results.
    candidates.topByKey($(k))(Ordering.by(-_._2))
  }

  /**
    * Copies [[ReebDiagramModel]] with extra parameters.
    */
  override def copy(extra: ParamMap): ReebDiagramModel = {
    val copied = new ReebDiagramModel(uid, topTrees, subTrees)
    copyValues(copied, extra).setParent(parent)
  }
}

object ReebDiagramModel {

  /**
    * Searches leaf indices for querying kNN search.
    *
    * @param key cover id and query vector pair to be searched
    * @param topTrees top-level tree used to index data points
    * @param bufferSize reluxing distance buffer width
    * @return indices to be searched for the query vector
    */
  private[feature] def search(key: (Int, VectorWithId),
                              topTrees: TopTrees,
                              bufferSize: Double): Seq[Int] = {
    def loop(query: VectorWithId, tree: Tree, acc: Int): Seq[Int] =
      tree match {
        case node: Branch =>
          val buf = new ArrayBuffer[Int]
          val d = node.measure(node.vantagePoint.vector, query.vector)
          if (d < node.vantagePoint.median) {
            buf ++= loop(query, node.left, acc)
            if (bufferSize > math.abs(d - node.vantagePoint.median))
              buf ++= loop(query, node.right, acc + node.left.numLeaves)
          } else {
            buf ++= loop(query, node.right, acc + node.left.numLeaves)
            if (bufferSize > math.abs(d - node.vantagePoint.median))
              buf ++= loop(query, node.left, acc)
          }
          buf
        case _ => Seq(acc)
      }
    val (entry, query) = (topTrees.trees.get(key._1), key._2)
    require(entry.isDefined,
            s"undefined cover identifier is passed: ${key._1}")
    loop(query, entry.get.tree, entry.get.indexStartsFrom)
  }

  /**
    * Represents vertex attributes for Ertoez's algorithm.
    *
    * @param knn a hamming encoded k-kearest neighbor representation of a data point
    * @param nearestNeighbor the nearest neighbor's id of the vertex in hamming space in terms of hamming similarity
    * @param density
    */
  private[feature] final case class VertexAttr(val knn: OldSparseVector,
                                               val nearestNeighbor: VertexId,
                                               val density: Int)

  /** Represents intermediate graph type for Ertoez's algorithm. */
  private[feature] type ErtoezGraph = Graph[VertexAttr, Double]

  /** Computes Ertoes similarity. */
  private[feature] def ErtoezSimilarity(lhs: OldVector,
                                        rhs: OldVector): Double = {
    val (li, ri) = (lhs.toSparse.indices.toSet, rhs.toSparse.indices.toSet)
    (li intersect ri).size.toDouble
  }

  /**
    * GraphX/Pregel Ertoez’s SNN-DBSCAN implementation for calculating the connected components of a graph.
    * This implementation is based on the following report:
    * [[https://pdfs.semanticscholar.org/6e96/5e32fa8e69eb4f16175745aef956fc332d43.pdf
    *   Finding Topics in Collections of Documents: A Shared Nearest Neighbor Approach]]
    *
    * @param kNNs kNNs for each data points
    * @param k number of k of k nearest neighbors
    * @param linkThresholdRatio the weight coefficient to the number of neighbors which determines the edge is
    *                           linked or not.
    * @param coreThresholdRatio the weight coefficient to the number of neighbors which determines the data point is
    *                           core point or not
    * @return Ertoes graph
    * @see https://pdfs.semanticscholar.org/6e96/5e32fa8e69eb4f16175745aef956fc332d43.pdf
    */
  private[feature] def toErtoezGraph(
      kNNs: RDD[(Long, Array[(Long, Double)])],
      k: Int,
      linkThresholdRatio: Double,
      coreThresholdRatio: Double): ErtoezGraph = {
    // Cluculate thresholds.
    val linkThreshold = k.toDouble * linkThresholdRatio
    val coreThreshold = k.toDouble * coreThresholdRatio
    // Construct similarity matrix.
    val similarityMatrix = new CoordinateMatrix(
      kNNs.flatMap {
        case (i, neighbors) =>
          neighbors.map {
            case (j, _) =>
              new MatrixEntry(i, j, 1.0)
          }
      }
    )
    // Constructs influence matrix.
    val influenceMatrix =
      similarityMatrix.hproduct(similarityMatrix.transpose(), (_ => 1.0))
    // Initialize vertices.
    val vertices = similarityMatrix.toIndexedRowMatrix.rows
      .map(row => (row.index, VertexAttr(row.vector.toSparse, 0L, 0)))
    // Initialize edges.
    val edges = influenceMatrix.entries
      .map(e => Edge(e.i, e.j, e.value))
    // Constructs shared nearest neighbor graph.
    val sharedNearestNeighbors = Graph(vertices, edges)
      .mapTriplets(t => ErtoezSimilarity(t.srcAttr.knn, t.dstAttr.knn))
    // Computes nearest neghbors in according to the SNN similarity
    val nearestNeighbors = sharedNearestNeighbors
      .aggregateMessages[(VertexId, Double, Int)](
        ctx => {
          ctx.sendToSrc(
            (ctx.dstId, ctx.attr, if (ctx.attr > linkThreshold) 1 else 0))
        }, {
          case (lhs, rhs) =>
            val (nearestNeighbor, similarity) =
              if (lhs._2 > rhs._2) (lhs._1, lhs._2) else (rhs._1, rhs._2)
            (nearestNeighbor, similarity, lhs._3 + rhs._3)
        }
      )
      .map {
        case (i, (nearestNeighbor, _, density)) =>
          (i, (nearestNeighbor, density))
      }
    // Assigns nearest neighbor ids to each vertices.
    def isCorePoint(e: EdgeTriplet[VertexAttr, Double]): Boolean =
      e.attr > linkThreshold && e.srcAttr.density > coreThreshold
    def isBoundary(e: EdgeTriplet[VertexAttr, Double]): Boolean =
      e.attr > linkThreshold && (e.srcAttr.nearestNeighbor == e.dstId && e.dstAttr.density > coreThreshold)
    def removeIsolatedVertices[VD: ClassTag, ED: ClassTag](
        graph: Graph[VD, ED]): Graph[VD, ED] =
      graph.filter[(VD, Boolean), ED](
        g =>
          g.outerJoinVertices[Int, (VD, Boolean)](graph.degrees) {
            (_, attr, degree) =>
              (attr, degree.isDefined)
        },
        vpred = (v, attr) => attr._2
      )
    val corePoints = sharedNearestNeighbors
      .outerJoinVertices(nearestNeighbors) {
        case (i, attr, nearestNeighbor) =>
          val (nearestNeighborId, density) =
            nearestNeighbor.getOrElse((i, 0))
          VertexAttr(attr.knn, nearestNeighborId, density)
      }
      .subgraph(
        epred = (e => { isCorePoint(e) || isBoundary(e) })
      )
      .cache()
    removeIsolatedVertices(corePoints)
  }
}
