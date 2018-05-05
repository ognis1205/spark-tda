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

import java.util.{Random => JRandom}
import scala.reflect.ClassTag
import org.scalacheck.{Arbitrary, Gen}
import org.scalacheck.Arbitrary.arbitrary
import org.scalacheck.Gen.{choose, listOfN}
import org.scalatest.PropSpec
import org.apache.spark.mllib.linalg.{Vector, SparseVector, DenseVector}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import com.holdenkarau.spark.testing.SharedSparkContext

/**
  * PropSpec base for impl package.
  *
  * @author Shingo OKAWA
  */
abstract class ImplPropSpec extends PropSpec with SharedSparkContext {
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

  private def chooseDim(featureUpper: Int, signatureUpper: Int) =
    for {
      numFeatures <- choose(1, featureUpper)
      signatureLength <- choose(1, signatureUpper)
    } yield (numFeatures, signatureLength)

  private def isPrime(x: Int): Boolean =
    (x > 1) && !(2 to scala.math.sqrt(x).toInt).exists(y => x % y == 0)

  private def choosePrime(upper: Int) =
    for {
      prime <- choose(2, upper) suchThat (isPrime(_))
    } yield prime

  private def chooseBucket(upper: Double) =
    for {
      bucket <- choose(0.0, upper) suchThat (_ > 0.0)
    } yield bucket

  private def vectorOfN(length: Int, gen: Gen[Double]): Gen[DenseVector] =
    listOfN(length, gen).map { list =>
      new DenseVector(list.toArray)
    }

  val simhashGen = for {
    (numFeatures, signatureLength) <- chooseDim(10, 10)
    vector <- vectorOfN(numFeatures, choose(-10.0, 10.0))
  } yield (vector, signatureLength, SimHash(numFeatures, signatureLength))

  val minhashGen = for {
    (numFeatures, signatureLength) <- chooseDim(10, 10)
    prime <- choosePrime(7)
    vector <- vectorOfN(numFeatures, choose(-10.0, 10.0))
  } yield
    (vector, signatureLength, MinHash(numFeatures, signatureLength, prime))

  val pstableGen = for {
    (numFeatures, signatureLength) <- chooseDim(10, 10)
    bucketWidth <- chooseBucket(10.0)
    vector <- vectorOfN(numFeatures, choose(-10.0, 10.0))
  } yield
    (vector,
     signatureLength,
     PStable.L1(numFeatures, signatureLength, bucketWidth),
     PStable.L2(numFeatures, signatureLength, bucketWidth))

  val bsampleGen = for {
    (numFeatures, signatureLength) <- chooseDim(10, 10)
    vector <- vectorOfN(numFeatures, choose(-10.0, 10.0))
  } yield (vector, signatureLength, BitSampling(numFeatures, signatureLength))

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

  private def simhashTable(numFeatures: Int,
                           signatureLength: Int): Gen[Seq[(Int, SimHash)]] =
    for {
      numTables <- choose(1, 5)
    } yield
      (0 until numTables).map(table =>
        (table, SimHash(numFeatures, signatureLength)))

  private def minhashTable(numFeatures: Int,
                           signatureLength: Int): Gen[Seq[(Int, MinHash)]] =
    for {
      numTables <- choose(1, 5)
      prime <- choosePrime(7)
    } yield
      (0 until numTables).map(table =>
        (table, MinHash(numFeatures, signatureLength, prime)))

  val simhashBucketsGen = for {
    (numFeatures, signatureLength) <- chooseDim(10, 10)
    numVectors <- choose(1, 100)
    simhashes <- simhashTable(numFeatures, signatureLength)
    vectors <- vectorsOfNM(numVectors, numFeatures, choose(-10.0, 10.0))
  } yield {
    (sc.parallelize(vectors.zipWithIndex.flatMap {
      case (vector, i) =>
        simhashes.map {
          case (table, simhash) =>
            simhash(i.toLong, table, vector)
        }
    }), numVectors)
  }

  val minhashBucketsGen = for {
    (numFeatures, signatureLength) <- chooseDim(10, 10)
    numVectors <- choose(1, 100)
    numBands <- choose(1, 5)
    minhashes <- minhashTable(numFeatures, signatureLength)
    vectors <- vectorsOfNM(numVectors, numFeatures, choose(-10.0, 10.0))
  } yield {
    (sc.parallelize(vectors.zipWithIndex.flatMap {
      case (vector, i) =>
        minhashes.map {
          case (table, minhash) =>
            minhash(i.toLong, table, vector)
        }
    }), numVectors, numBands)
  }

  val simhashJoinGen = for {
    (numFeatures, signatureLength) <- chooseDim(10, 10)
    numVectors <- choose(1, 100)
    quantity <- choose(1, 10)
    smallTable <- choose(1, 5)
    bigTable <- choose(6, 10)
    vectors <- vectorsOfNM(numVectors, numFeatures, choose(-10.0, 10.0))
  } yield
    (new IndexedRowMatrix(sc.parallelize(vectors.zipWithIndex.map {
       case (vector, i) =>
         new IndexedRow(i, vector)
     })),
     SimHashJoin(quantity, numFeatures, signatureLength, smallTable),
     SimHashJoin(quantity, numFeatures, signatureLength, bigTable))

  val minhashJoinGen = for {
    (numFeatures, signatureLength) <- chooseDim(10, 10)
    numVectors <- choose(1, 100)
    quantity <- choose(1, 10)
    smallTable <- choose(1, 5)
    bigTable <- choose(6, 10)
    numBands <- choose(1, 5)
    prime <- choosePrime(7)
    vectors <- vectorsOfNM(numVectors, numFeatures, choose(-10.0, 10.0))
  } yield
    (new IndexedRowMatrix(sc.parallelize(vectors.zipWithIndex.map {
       case (vector, i) =>
         new IndexedRow(i, vector)
     })),
     MinHashJoin(quantity,
                 numFeatures,
                 signatureLength,
                 smallTable,
                 numBands,
                 prime),
     MinHashJoin(quantity,
                 numFeatures,
                 signatureLength,
                 bigTable,
                 numBands,
                 prime))

  val pstablel1JoinGen = for {
    (numFeatures, signatureLength) <- chooseDim(10, 10)
    numVectors <- choose(1, 100)
    quantity <- choose(1, 10)
    smallTable <- choose(1, 5)
    bigTable <- choose(6, 10)
    bucketWidth <- choose(0.0, 10.0) suchThat (_ > 0.0)
    vectors <- vectorsOfNM(numVectors, numFeatures, choose(-10.0, 10.0))
  } yield
    (new IndexedRowMatrix(sc.parallelize(vectors.zipWithIndex.map {
       case (vector, i) =>
         new IndexedRow(i, vector)
     })),
     PStableL1Join(quantity,
                   numFeatures,
                   signatureLength,
                   smallTable,
                   bucketWidth),
     PStableL1Join(quantity,
                   numFeatures,
                   signatureLength,
                   bigTable,
                   bucketWidth))

  val pstablel2JoinGen = for {
    (numFeatures, signatureLength) <- chooseDim(10, 10)
    numVectors <- choose(1, 100)
    quantity <- choose(1, 10)
    smallTable <- choose(1, 5)
    bigTable <- choose(6, 10)
    bucketWidth <- choose(0.0, 10.0) suchThat (_ > 0.0)
    vectors <- vectorsOfNM(numVectors, numFeatures, choose(-10.0, 10.0))
  } yield
    (new IndexedRowMatrix(sc.parallelize(vectors.zipWithIndex.map {
       case (vector, i) =>
         new IndexedRow(i, vector)
     })),
     PStableL2Join(quantity,
                   numFeatures,
                   signatureLength,
                   smallTable,
                   bucketWidth),
     PStableL2Join(quantity,
                   numFeatures,
                   signatureLength,
                   bigTable,
                   bucketWidth))

  val bsampleJoinGen = for {
    (numFeatures, signatureLength) <- chooseDim(10, 10)
    numVectors <- choose(1, 100)
    quantity <- choose(1, 10)
    smallTable <- choose(1, 5)
    bigTable <- choose(6, 10)
    vectors <- vectorsOfNM(numVectors, numFeatures, choose(-10.0, 10.0))
  } yield
    (new IndexedRowMatrix(sc.parallelize(vectors.zipWithIndex.map {
       case (vector, i) =>
         new IndexedRow(i, vector)
     })),
     BitSamplingJoin(quantity, numFeatures, signatureLength, smallTable),
     BitSamplingJoin(quantity, numFeatures, signatureLength, bigTable))

  val cosineJoinGen = for {
    (numFeatures, signatureLength) <- chooseDim(10, 10)
    numVectors <- choose(1, 100)
    quantity <- choose(1, 10)
    vectors <- vectorsOfNM(numVectors, numFeatures, choose(-10.0, 10.0))
  } yield
    (numVectors,
     quantity,
     new IndexedRowMatrix(sc.parallelize(vectors.zipWithIndex.map {
       case (vector, i) =>
         new IndexedRow(i, vector)
     })),
     CosineDistanceJoin(quantity))

  val jaccardJoinGen = for {
    (numFeatures, signatureLength) <- chooseDim(10, 10)
    numVectors <- choose(1, 100)
    quantity <- choose(1, 10)
    vectors <- vectorsOfNM(numVectors, numFeatures, choose(-10.0, 10.0))
  } yield
    (numVectors,
     quantity,
     new IndexedRowMatrix(sc.parallelize(vectors.zipWithIndex.map {
       case (vector, i) =>
         new IndexedRow(i, vector)
     })),
     JaccardDistanceJoin(quantity))

  val manhattanJoinGen = for {
    (numFeatures, signatureLength) <- chooseDim(10, 10)
    numVectors <- choose(1, 100)
    quantity <- choose(1, 10)
    vectors <- vectorsOfNM(numVectors, numFeatures, choose(-10.0, 10.0))
  } yield
    (numVectors,
     quantity,
     new IndexedRowMatrix(sc.parallelize(vectors.zipWithIndex.map {
       case (vector, i) =>
         new IndexedRow(i, vector)
     })),
     ManhattanDistanceJoin(quantity))

  val euclideanJoinGen = for {
    (numFeatures, signatureLength) <- chooseDim(10, 10)
    numVectors <- choose(1, 100)
    quantity <- choose(1, 10)
    vectors <- vectorsOfNM(numVectors, numFeatures, choose(-10.0, 10.0))
  } yield
    (numVectors,
     quantity,
     new IndexedRowMatrix(sc.parallelize(vectors.zipWithIndex.map {
       case (vector, i) =>
         new IndexedRow(i, vector)
     })),
     EuclideanDistanceJoin(quantity))

  val hammingJoinGen = for {
    (numFeatures, signatureLength) <- chooseDim(10, 10)
    numVectors <- choose(1, 100)
    quantity <- choose(1, 10)
    vectors <- vectorsOfNM(numVectors, numFeatures, choose(-10.0, 10.0))
  } yield
    (numVectors,
     quantity,
     new IndexedRowMatrix(sc.parallelize(vectors.zipWithIndex.map {
       case (vector, i) =>
         new IndexedRow(i, vector)
     })),
     HammingDistanceJoin(quantity))
}
