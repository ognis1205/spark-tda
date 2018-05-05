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
package org.apache.spark.ml.util.interval

import org.scalacheck.Gen
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.Matchers
import org.scalatest.prop.GeneratorDrivenPropertyChecks

/**
  * PropSpec for [[org.apache.spark.ml.util.interval]].
  *
  * @author Shingo OKAWA
  */
class PackageTest
    extends IntervalPropSpec
    with GeneratorDrivenPropertyChecks
    with Matchers {
  val coverGen = for {
    numberOfSplits <- Gen.posNum[Int]
    overlapRatio <- Gen.choose(0.0, 1.0) if overlapRatio > 0.0
    list <- Gen.listOfN(3, arbitrary[Double])
  } yield (numberOfSplits, overlapRatio, list)

  property("open cover covers all range of specified hypercube") {
    forAll(coverGen) {
      case (numberOfSplits, overlapRatio, list) =>
        val sorted = list.sorted
        val lower = sorted(0)
        val upper = sorted(2)
        val point = sorted(1)
        OpenCover(numberOfSplits, overlapRatio, Vector((lower, upper)))
          .foldLeft(false) { (acc, open) =>
            acc || (open contains Vector(point))
          } should be(true)
    }
  }

  property("closed cover covers all range of specified hypercube") {
    forAll(coverGen) {
      case (numberOfSplits, overlapRatio, list) =>
        val sorted = list.sorted
        val lower = sorted(0)
        val upper = sorted(2)
        val point = sorted(1)
        ClosedCover(numberOfSplits, overlapRatio, Vector((lower, upper)))
          .foldLeft(false) { (acc, closed) =>
            acc || (closed contains Vector(point))
          } should be(true)
    }
  }
}
