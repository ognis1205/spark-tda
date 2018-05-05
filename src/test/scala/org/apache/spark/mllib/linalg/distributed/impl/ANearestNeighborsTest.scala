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

import org.scalacheck.Prop.forAllNoShrink
import org.scalatest.Matchers
import org.scalatest.prop.GeneratorDrivenPropertyChecks

/**
  * FunSpec for [[org.apache.spark.mllib.linalg.distributed.impl.ANearestNeighbors]].
  *
  * @author Shingo OKAWA
  */
class ANearestNeighborsTest
    extends ImplPropSpec
    with GeneratorDrivenPropertyChecks
    with Matchers {

  property(
    "average selectivity of simhash joins increase with more hash tables") {
    forAllNoShrink(simhashJoinGen) {
      case (matrix, joinWithLessTables, joinWithMoreTables) =>
        val lessSelectivity = joinWithLessTables.avgSelectivity(matrix)
        val moreSelectivity = joinWithMoreTables.avgSelectivity(matrix)
        lessSelectivity <= moreSelectivity
    }
  }

  property(
    "average selectivity of minhash joins increase with more hash tables") {
    forAllNoShrink(minhashJoinGen) {
      case (matrix, joinWithLessTables, joinWithMoreTables) =>
        val lessSelectivity = joinWithLessTables.avgSelectivity(matrix)
        val moreSelectivity = joinWithMoreTables.avgSelectivity(matrix)
        lessSelectivity <= moreSelectivity
    }
  }

  property(
    "average selectivity of p-stable L1 joins increase with more hash tables") {
    forAllNoShrink(pstablel1JoinGen) {
      case (matrix, joinWithLessTables, joinWithMoreTables) =>
        val lessSelectivity = joinWithLessTables.avgSelectivity(matrix)
        val moreSelectivity = joinWithMoreTables.avgSelectivity(matrix)
        lessSelectivity <= moreSelectivity
    }
  }

  property(
    "average selectivity of p-stable L2 joins increase with more hash tables") {
    forAllNoShrink(pstablel2JoinGen) {
      case (matrix, joinWithLessTables, joinWithMoreTables) =>
        val lessSelectivity = joinWithLessTables.avgSelectivity(matrix)
        val moreSelectivity = joinWithMoreTables.avgSelectivity(matrix)
        lessSelectivity <= moreSelectivity
    }
  }

  property(
    "average selectivity of bit sampling joins increase with more hash tables") {
    forAllNoShrink(bsampleJoinGen) {
      case (matrix, joinWithLessTables, joinWithMoreTables) =>
        val lessSelectivity = joinWithLessTables.avgSelectivity(matrix)
        val moreSelectivity = joinWithMoreTables.avgSelectivity(matrix)
        lessSelectivity <= moreSelectivity
    }
  }
}
