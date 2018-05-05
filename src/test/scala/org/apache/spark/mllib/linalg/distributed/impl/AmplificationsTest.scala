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
package org.apache.spark.mllib.linalg.distributed.impl

import org.scalacheck.Gen.{choose, oneOf, listOfN}
import org.scalacheck.Arbitrary.arbitrary
import org.scalacheck.Prop.forAllNoShrink
import org.scalatest.Matchers
import org.scalatest.prop.GeneratorDrivenPropertyChecks
import org.apache.spark.mllib.linalg.{
  DenseVector,
  CosineDistance,
  JaccardDistance
}
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix

/**
  * FunSpec for [[org.apache.spark.mllib.linalg.distributed.impl.Amplifications]].
  *
  * @author Shingo OKAWA
  */
class AmplificationsTest
    extends ImplPropSpec
    with GeneratorDrivenPropertyChecks
    with Matchers {

  property(
    "or-construction generates the correct number of indexed rows for the given data points") {
    forAllNoShrink(simhashBucketsGen) {
      case (buckets, numVectors) =>
        val or = ORConstruction(CosineDistance)
        val sim =
          new CoordinateMatrix(or(buckets, numVectors)).toIndexedRowMatrix.rows
            .collect()
        sim.size === numVectors
        sim.forall(s => s.vector.size <= numVectors)
    }
  }

  property(
    "band or-construction generates the correct number of indexed rows for the given data points") {
    forAllNoShrink(minhashBucketsGen) {
      case (buckets, numVectors, numBands) =>
        val bor = BandORConstruction(JaccardDistance, numBands)
        val sim =
          new CoordinateMatrix(bor(buckets, numVectors)).toIndexedRowMatrix.rows
            .collect()
        sim.size === numVectors
        sim.forall(s => s.vector.size <= numVectors)
    }
  }
}
