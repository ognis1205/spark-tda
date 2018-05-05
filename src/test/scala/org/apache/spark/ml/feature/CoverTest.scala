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

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions.{col, explode, udf}
import org.scalatest.{PropSpec, Matchers, GivenWhenThen}
import org.scalatest.prop.GeneratorDrivenPropertyChecks

/**
  * PropSpec for [[org.apache.spark.ml.feature.Cover]].
  *
  * @author Shingo OKAWA
  */
class CoverTest
    extends FeaturePropSpec
    with GivenWhenThen
    with GeneratorDrivenPropertyChecks
    with Matchers {
  val assembler = new VectorAssembler()
    .setInputCols(Array("double", "integer"))
    .setOutputCol("vector")

  property("argument numSplits must be positive") {
    intercept[IllegalArgumentException] {
      val cover = new Cover()
        .setInputCols("double")
        .setOutputCol("cover_ids")
        .setNumSplits(0)
    }
  }

  property("argument overlapRatio must be positive") {
    intercept[IllegalArgumentException] {
      val cover = new Cover()
        .setInputCols("double")
        .setOutputCol("cover_ids")
        .setOverlapRatio(0.0)
    }
  }

  property("cover estimator changes nothing with the original dataframe") {
    val cover = new Cover()
      .setInputCols("double", "integer", "vector")
      .setOutputCol("cover_ids")

    forAll(dataframeGen.arbitrary) { df =>
      val transformed = assembler.transform(df)
      whenever(
        transformed.count() > 0 && hasDistinctValues(transformed,
                                                     "double",
                                                     "integer",
                                                     "vector")) {
        val covered = cover
          .fit(transformed)
          .transform(transformed)
          .drop("cover_ids")
          .except(transformed)
          .count() should be(0)
      }
    }
  }

  property("generated cover covers all range of specified columns") {
    val cover = new Cover()
      .setInputCols("double", "integer", "vector")
      .setOutputCol("cover_ids")
    val uncovered = udf { xs: Seq[Long] =>
      xs.length == 0
    }

    forAll(dataframeGen.arbitrary) { df =>
      val transformed = assembler.transform(df)
      whenever(
        transformed.count() > 0 && hasDistinctValues(transformed,
                                                     "double",
                                                     "integer",
                                                     "vector")) {
        cover
          .fit(transformed)
          .transform(transformed)
          .where(uncovered(col("cover_ids")))
          .count() should be(0)
      }
    }
  }

  property("Cover is readable/writable") {
    val cover = new Cover()
      .setInputCols("double", "integer")
      .setOutputCol("cover_ids")
    testDefaultReadWrite(cover)
  }

  property("CoverModel is readable/writable") {
    val model = new CoverModel("myCoverModel",
                               Vectors.dense(-1.0, 0.0),
                               Vectors.dense(1.0, 10.0))
      .setInputCols("double", "integer")
      .setOutputCol("cover_ids")
    val newModel = testDefaultReadWrite(model)
    assert(newModel.min === model.min)
    assert(newModel.max === model.max)
  }
}
