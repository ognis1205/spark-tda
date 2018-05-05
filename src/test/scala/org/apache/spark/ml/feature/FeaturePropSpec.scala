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

import org.apache.spark.ml.linalg.{Vector, Vectors, DenseVector}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.types.{
  StructField,
  IntegerType,
  DoubleType,
  BooleanType,
  StructType,
  StringType,
  ArrayType
}
import org.scalacheck.{Arbitrary, Gen}
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.PropSpec
import com.holdenkarau.spark.testing.{
  SharedSparkContext,
  DataframeGenerator,
  Column
}

/**
  * PropSpec base for feature package.
  *
  * @author Shingo OKAWA
  */
abstract class FeaturePropSpec
    extends PropSpec
    with SharedSparkContext
    with DefaultReadWriteTest {
  implicit def arbitraryDenseVector: Arbitrary[DenseVector] =
    Arbitrary {
      for (arr <- arbitrary[Array[Double]]) yield new DenseVector(arr)
    }

  implicit def arbitraryVector: Arbitrary[Vector] =
    Arbitrary(
      Gen.frequency(
        1 -> arbitrary[DenseVector]
      ))

  lazy val spark = SparkSession.builder().getOrCreate()

  def schema =
    StructType(
      List(
        StructField("integer", IntegerType),
        StructField("double", DoubleType),
        StructField("boolean", BooleanType),
        StructField("string", StringType)
      ))

  def integerGen = new Column("integer", Gen.choose(-100, 100))

  def doubleGen = new Column("double", Gen.choose(-100.0, 100.0))

  def stringGen =
    new Column("string", Gen.oneOf("A", "BC", "DEF", "GHIJ", "KLMNO"))

  def dataframeGen =
    DataframeGenerator.arbitraryDataFrameWithCustomFields(
      spark.sqlContext,
      schema)(integerGen, doubleGen, stringGen)

  def hasDistinctValues(df: DataFrame, columns: String*): Boolean = {
    columns.foldLeft(true) { (acc, col) =>
      acc && df.select(col).distinct.count() > 1
    }
  }
}
