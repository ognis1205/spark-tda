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
package org.apache.spark.ml.linalg

import org.scalatest._

/**
  * FunSpec for [[org.apache.spark.ml.linalg.Metrics]].
  *
  * @author Shingo OKAWA
  */
class MetricsTest extends LinalgFunSuite {
  import org.scalactic.Tolerance._
  val sv0 = new SparseVector(10, Array(0, 3, 6, 8), Array(1.0, 1.0, 1.0, 1.0))
  val sv1 = new SparseVector(10, Array(1, 4, 7, 9), Array(1.0, 1.0, 1.0, 1.0))
  val sv2 = new SparseVector(10, Array(2, 5, 7, 9), Array(1.0, 1.0, 1.0, 1.0))
  val dv0 = new DenseVector(
    Array(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0))
  val dv1 = new DenseVector(
    Array(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0))
  val dv2 = new DenseVector(
    Array(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0))

  test("cosine distance returns expected values") {
    assert(CosineDistance(sv0, sv0) === 0.0)
    assert(CosineDistance(sv0, sv1) === 1.0)
    assert(CosineDistance(sv1, sv2) === 0.5)
    assert(CosineDistance(dv0, dv0) === 0.0)
    assert(CosineDistance(dv0, dv1) === 1.0)
    assert(CosineDistance(dv1, dv2) === 0.5)
    assert(CosineDistance(sv0, dv0) === 0.0)
    assert(CosineDistance(sv0, dv1) === 1.0)
    assert(CosineDistance(sv1, dv2) === 0.5)
  }

  test("euclidean distance returns expected values") {
    assert(EuclideanDistance(sv0, sv0) === 0.0)
    assert(EuclideanDistance(sv0, sv1) === 2.83 +- 0.01)
    assert(EuclideanDistance(sv1, sv2) === 2.0)
    assert(EuclideanDistance(dv0, dv0) === 0.0)
    assert(EuclideanDistance(dv0, dv1) === 2.83 +- 0.01)
    assert(EuclideanDistance(dv1, dv2) === 2.0)
    assert(EuclideanDistance(sv0, dv0) === 0.0)
    assert(EuclideanDistance(sv0, dv1) === 2.83 +- 0.01)
    assert(EuclideanDistance(sv1, dv2) === 2.0)
  }

  test("manhattan distance returns expected values") {
    assert(ManhattanDistance(sv0, sv0) === 0.0)
    assert(ManhattanDistance(sv0, sv1) === 8.0)
    assert(ManhattanDistance(sv1, sv2) === 4.0)
    assert(ManhattanDistance(dv0, dv0) === 0.0)
    assert(ManhattanDistance(dv0, dv1) === 8.0)
    assert(ManhattanDistance(dv1, dv2) === 4.0)
    assert(ManhattanDistance(sv0, dv0) === 0.0)
    assert(ManhattanDistance(sv0, dv1) === 8.0)
    assert(ManhattanDistance(sv1, dv2) === 4.0)
  }

  test("hamming distance returns expected values") {
    assert(HammingDistance(sv0, sv0) === 0.0)
    assert(HammingDistance(sv0, sv1) === 8.0)
    assert(HammingDistance(sv1, sv2) === 4.0)
    assert(HammingDistance(dv0, dv0) === 0.0)
    assert(HammingDistance(dv0, dv1) === 8.0)
    assert(HammingDistance(dv1, dv2) === 4.0)
    assert(HammingDistance(sv0, dv0) === 0.0)
    assert(HammingDistance(sv0, dv1) === 8.0)
    assert(HammingDistance(sv1, dv2) === 4.0)
  }

  test("jaccard distance returns expected values") {
    assert(JaccardDistance(sv0, sv0) === 0.0)
    assert(JaccardDistance(sv0, sv1) === 1.0)
    assert(JaccardDistance(sv1, sv2) === 0.67 +- 0.01)
    assert(JaccardDistance(dv0, dv0) === 0.0)
    assert(JaccardDistance(dv0, dv1) === 1.0)
    assert(JaccardDistance(dv1, dv2) === 0.67 +- 0.01)
    assert(JaccardDistance(sv0, dv0) === 0.0)
    assert(JaccardDistance(sv0, dv1) === 1.0)
    assert(JaccardDistance(sv1, dv2) === 0.67 +- 0.01)
  }
}
