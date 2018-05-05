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

import scala.reflect.ClassTag
import org.scalacheck.{Arbitrary, Gen}
import org.scalacheck.Arbitrary.arbitrary
import org.scalacheck.Gen.{choose, oneOf}
import org.scalatest.PropSpec
import org.apache.spark.ml.linalg.{
  CosineDistance,
  EuclideanDistance,
  ManhattanDistance,
  JaccardDistance,
  HammingDistance
}
import org.apache.spark.ml.linalg.{Vector, SparseVector, DenseVector, Vectors}
import com.holdenkarau.spark.testing.SharedSparkContext

/**
  * PropSpec base for knn package.
  *
  * @author Shingo OKAWA
  */
abstract class KNNPropSpec extends PropSpec with SharedSparkContext {
  implicit def arbitraryDenseVector: Arbitrary[DenseVector] =
    Arbitrary {
      for (arr <- arbitrary[Array[Double]]) yield new DenseVector(arr)
    }

  implicit def arbitrarySparseVector: Arbitrary[SparseVector] =
    Arbitrary {
      for (vec <- arbitrary[DenseVector]) yield vec.toSparse
    }

  implicit def arbitraryVector: Arbitrary[Vector] =
    Arbitrary(
      Gen.frequency(
        1 -> arbitrary[DenseVector],
        1 -> arbitrary[SparseVector]
      ))

  private def arraysOfNM[T: ClassTag](numRows: Int,
                                      numCols: Int,
                                      gen: Gen[T]): Gen[Array[Array[T]]] =
    Gen.listOfN(numRows * numCols, gen).map { square =>
      square.toArray.grouped(numCols).toArray
    }

  private def vectorsOfNM(numRows: Int,
                          numCols: Int,
                          gen: Gen[Double]): Gen[Array[DenseVector]] =
    for {
      arrays <- arraysOfNM(numRows, numCols, gen)
    } yield arrays.map(arr => new DenseVector(arr))

  val treeGen = for {
    measure <- oneOf(CosineDistance,
                     EuclideanDistance,
                     ManhattanDistance,
                     HammingDistance,
                     JaccardDistance)
    numVectors <- choose(1, 100)
    vectors <- vectorsOfNM(numVectors, 2, choose(-10.0, 10.0))
  } yield
    vectors
      .scanLeft(Seq[Vector]())(_ :+ _)
      .tail
      .map(
        vs =>
          VPTree(vs.map(v => VectorEntry(0L, v)).toIndexedSeq,
                 measure,
                 10,
                 10,
                 10))
}
