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
package org.apache.spark.ml.util.knn

import org.scalacheck.Prop.forAllNoShrink
import org.scalatest.Matchers
import org.scalatest.prop.GeneratorDrivenPropertyChecks
import org.apache.spark.ml.linalg.{Vector, Vectors, EuclideanDistance}

/**
  * FunSpec for [[org.apache.spark.ml.util.knn.Partitioners]].
  *
  * @author Shingo OKAWA
  */
class PartitionersTest
    extends KNNPropSpec
    with GeneratorDrivenPropertyChecks
    with Matchers {
  property("TopTreesPartitioner can be constructed with empty data") {
    forAll { (v: Vector, coverId: Int) =>
      val partitioner =
        new TopTreesPartitioner(TopTrees(IndexedSeq.empty[(Int, Tree)]))
      val vector = VectorEntry(0L, v)
      intercept[NoSuchElementException] {
        partitioner.getPartition((coverId, vector))
      }
    }
  }

  property(
    "TopTrees can be constructed with non empty data and maintain its consistency") {
    forAll(treeGen) {
      case (trees) =>
        val indexedTrees = trees.zipWithIndex.map { case (t, i) => (i, t) }
        val partitioner = new TopTreesPartitioner(TopTrees(indexedTrees))
        val indices = indexedTrees
          .flatMap {
            case (index, tree) =>
              tree.iterator.map(d => (index, d))
          }
          .map {
            case (index, entry) =>
              partitioner.getPartition((index, entry))
          }
          .toSet
        indices should contain theSameElementsAs (0 until partitioner.numPartitions)
          .toSet
        (0 until partitioner.numPartitions).toSet should contain theSameElementsAs indices
        intercept[IllegalArgumentException] {
          partitioner.getPartition(0)
        }
    }
  }
}
