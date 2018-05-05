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
package org.apache.spark.mllib.linalg

import java.util.Random
import org.scalacheck.Gen
import org.scalacheck.Prop.forAllNoShrink
import org.scalatest.Matchers
import org.scalatest.prop.GeneratorDrivenPropertyChecks
import org.scalatest.prop.Checkers.check

/**
  * PropSpec for [[org.apache.spark.mllib.linalg.Projections]].
  *
  * @author Shingo OKAWA
  */
class ProjectionsTest
    extends LinalgPropSpec
    with GeneratorDrivenPropertyChecks
    with Matchers {
  import org.scalactic.Tolerance._
  val dimGen = for {
    srcDim <- Gen.choose(100, 200)
    dstDim <- Gen.choose(100, 200)
  } yield (srcDim, dstDim)

  property("gaussian random projection have good statistical properties") {
    forAllNoShrink(dimGen) {
      case (srcDim, dstDim) =>
        val projection =
          GaussianRandomProjection(srcDim, dstDim, new Random())
        projection.mean === 0.0 +- 0.01
        projection.stddev === 1.0 +- 0.01
    }
  }

  property("cauchy random projection have good statistical properties") {
    forAllNoShrink(dimGen) {
      case (srcDim, dstDim) =>
        val projection =
          CauchyRandomProjection(srcDim, dstDim)
        projection.median === 0.0 +- 0.01
    }
  }
}
