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
package org.apache.spark.mllib.linalg.distributed

import org.scalacheck.Prop.forAllNoShrink
import org.scalatest.Matchers
import org.scalatest.prop.GeneratorDrivenPropertyChecks
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrixFunctions._

/**
  * FunSpec for [[org.apache.spark.mllib.linalg.distributed.CoordinateMatrixFunctions]].
  *
  * @author Shingo OKAWA
  */
class CoordinatMatrixFunctionsTest
    extends DistributedPropSpec
    with GeneratorDrivenPropertyChecks
    with Matchers {
  import org.scalactic.Tolerance._

  property("hadamard product preserved the size of matrices") {
    forAllNoShrink(coordinateMatrixGen) {
      case (lhs, rhs) =>
        val hprod = lhs hproduct rhs
        hprod.numRows == math.max(lhs.numRows, rhs.numRows)
        hprod.numCols == math.max(lhs.numCols, rhs.numCols)
    }
  }

  property(
    "resulting elements of hadamard product are equal to the product of corresponding elements") {
    forAllNoShrink(coordinateMatrixGen) {
      case (lhs, rhs) =>
        val lhsb = lhs.toBreeze
        val rhsb = rhs.toBreeze
        val hprod = (lhs hproduct rhs).toBreeze
        var assertion = true
        for (i <- (0 until math.min(lhs.numRows.toInt, rhs.numRows.toInt));
             j <- (0 until math.max(lhs.numCols.toInt, rhs.numCols.toInt))) {
          assertion &= (hprod(i, j) == lhsb(i, j) * rhsb(i, j) +- 0.01)
        }
        assertion
    }
  }
}
