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
  * FunSpec for [[org.apache.spark.ml.util.knn.Trees]].
  *
  * @author Shingo OKAWA
  */
class TreesTest
    extends KNNPropSpec
    with GeneratorDrivenPropertyChecks
    with Matchers {

  property("VPTree can be constructed with empty data") {
    forAll { (v: Vector) =>
      val tree =
        VPTree(IndexedSeq.empty[VectorWithId], EuclideanDistance, 0, 0)
      val vector = VectorEntry(0L, v)
      tree.iterator shouldBe empty
      tree.query(vector) shouldBe empty
      tree.numLeaves shouldBe 0
    }
  }

  property("VPTree can be constructed with data not having any duplication") {
    val origin = VectorEntry(0L, Vectors.dense(0, 0))
    val data = (-5 to 5).flatMap { i =>
      (-5 to 5).map { j =>
        VectorEntry(0L, Vectors.dense(i, j))
      }
    }
    List(1, data.size / 2, data.size, data.size * 2).foreach { leafSize =>
      val tree = VPTree(data, EuclideanDistance, 1, 1, leafSize)
      tree.size shouldBe data.size
      tree.iterator.toIterable should contain theSameElementsAs data
      data.foreach(v => tree.query(v, 1).head._1 shouldBe v)
      tree
        .query(origin, 5)
        .map(_._1.vector) should contain theSameElementsAs Set(
        Vectors.dense(-1, 0),
        Vectors.dense(1, 0),
        Vectors.dense(0, -1),
        Vectors.dense(0, 1),
        Vectors.dense(0, 0)
      )
      tree
        .query(origin, 9)
        .map(_._1.vector) should contain theSameElementsAs Set(
        Vectors.dense(-1, -1),
        Vectors.dense(-1, 0),
        Vectors.dense(-1, 1),
        Vectors.dense(0, -1),
        Vectors.dense(0, 0),
        Vectors.dense(0, 1),
        Vectors.dense(1, -1),
        Vectors.dense(1, 0),
        Vectors.dense(1, 1)
      )
      tree.numLeaves shouldBe (tree.cardinality / leafSize.toDouble).ceil
    }
  }

  property("VPTree can be constructed with data having duplication") {
    val origin = VectorEntry(0L, Vectors.dense(0, 0))
    val data =
      (Vectors.dense(2.0, 0.0) +: Array.fill(5)(Vectors.dense(0.0, 1.0)))
        .map(VectorEntry(0L, _))
    val tree = VPTree(data, EuclideanDistance, 6, 6)
    val knn = tree.query(origin, 5)
    tree.numLeaves shouldBe 2
    knn.size shouldBe 5
    knn.map(_._1.vector).toSet should contain theSameElementsAs Array(
      Vectors.dense(0.0, 1.0))
  }
}
