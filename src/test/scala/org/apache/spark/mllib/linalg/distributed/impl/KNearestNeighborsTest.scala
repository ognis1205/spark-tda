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
class KNearestNeighborsTest
    extends ImplPropSpec
    with GeneratorDrivenPropertyChecks
    with Matchers {

  property(
    "cosine distance join joins each feature vectors to its k-nearest neighbors") {
    forAllNoShrink(cosineJoinGen) {
      case (numVectors, quantity, matrix, join) =>
        val knn = join(matrix).toIndexedRowMatrix
        knn.rows.aggregate(true)(
          (acc, row) => acc && row.vector.toSparse.indices.size == quantity,
          _ && _)
        knn.rows.count == numVectors
    }
  }

  property(
    "jaccard distance join joins each feature vectors to its k-nearest neighbors") {
    forAllNoShrink(jaccardJoinGen) {
      case (numVectors, quantity, matrix, join) =>
        val knn = join(matrix).toIndexedRowMatrix
        knn.rows.aggregate(true)(
          (acc, row) => acc && row.vector.toSparse.indices.size == quantity,
          _ && _)
        knn.rows.count == numVectors
    }
  }

  property(
    "manhattan distance join joins each feature vectors to its k-nearest neighbors") {
    forAllNoShrink(manhattanJoinGen) {
      case (numVectors, quantity, matrix, join) =>
        val knn = join(matrix).toIndexedRowMatrix
        knn.rows.aggregate(true)(
          (acc, row) => acc && row.vector.toSparse.indices.size == quantity,
          _ && _)
        knn.rows.count == numVectors
    }
  }

  property(
    "euclidean distance join joins each feature vectors to its k-nearest neighbors") {
    forAllNoShrink(euclideanJoinGen) {
      case (numVectors, quantity, matrix, join) =>
        val knn = join(matrix).toIndexedRowMatrix
        knn.rows.aggregate(true)(
          (acc, row) => acc && row.vector.toSparse.indices.size == quantity,
          _ && _)
        knn.rows.count == numVectors
    }
  }

  property(
    "hamming distance join joins each feature vectors to its k-nearest neighbors") {
    forAllNoShrink(hammingJoinGen) {
      case (numVectors, quantity, matrix, join) =>
        val knn = join(matrix).toIndexedRowMatrix
        knn.rows.aggregate(true)(
          (acc, row) => acc && row.vector.toSparse.indices.size == quantity,
          _ && _)
        knn.rows.count == numVectors
    }
  }
}
