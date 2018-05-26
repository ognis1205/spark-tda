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

import org.apache.spark.ml.linalg.{Vectors, EuclideanDistance, Vector}
import org.apache.spark.sql.functions.{col, explode, udf}
import org.scalatest.{PropSpec, Matchers, GivenWhenThen}
import org.scalatest.prop.GeneratorDrivenPropertyChecks

/**
  * PropSpec for [[org.apache.spark.ml.feature.ReebDiagram]].
  *
  * @author Shingo OKAWA
  */
class ReebDiagramTest
    extends FeaturePropSpec
    with GivenWhenThen
    with GeneratorDrivenPropertyChecks
    with Matchers {
  val assembler = new VectorAssembler()
    .setInputCols(Array("double", "integer"))
    .setOutputCol("vector")
  val cover = new Cover()
    .setExploding(true)
    .setInputCols("double", "integer")
    .setOutputCol("cover_id")

  property("argument topTreeSize must be positive") {
    intercept[IllegalArgumentException] {
      val reeb = new ReebDiagram()
//        .setIdCol("id")
//        .setCoverCol("cover_id")
//        .setFeaturesCol("vector")
//        .setOutputCol("cluster_id")
        .setTopTreeSize(0)
    }
  }

  property("placeholder") {
    val reeb = new ReebDiagram()
      .setK(15)
      .setIdCol("id")
      .setCoverCol("cover_id")
      .setFeaturesCol("vector")
      .setOutputCol("cluster_id")
    forAll(dataframeGen.arbitrary) { df =>
      val assembled = assembler.transform(df)
      whenever(
        assembled.count() > 0 && hasDistinctValues(assembled,
                                                   "double",
                                                   "integer")) {
        val transformed = cover
          .fit(assembled)
          .transform(assembled)
        val result = reeb
          .setTopTreeSize(1)
          .fit(transformed)
          .transform(transformed)
//        result.show()
      }
    }
  }
}
