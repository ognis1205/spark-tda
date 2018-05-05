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

import java.util.{Random => JRandom}
import org.apache.spark.mllib.linalg.{Vector, Distance}
import org.apache.spark.mllib.linalg.distributed.impl.{
  Amplification,
  HashFunction,
  BucketEntry,
  Digest,
  SimHashJoin,
  MinHashJoin,
  PStableL1Join,
  PStableL2Join,
  BitSamplingJoin,
  CosineDistanceJoin,
  JaccardDistanceJoin,
  ManhattanDistanceJoin,
  EuclideanDistanceJoin,
  HammingDistanceJoin
}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.rdd.RDD

/**
  * Joins specified feature vectors with itself in accordance to the similarities. The resulting similarity matrix
  * is represented in the form of [[IndexedRowMatrix]].
  *
  * @author Shingo OKAWA
  */
trait SimilarityJoin extends Serializable {

  /** The maximum number of data points to be joined. */
  def quantity: Int

  /** The associating distance measure of this joiner. */
  def distance: Distance

  /** The storage level when the buckets persists. */
  def storageLevel: StorageLevel

  /**
    * Finds the "nearest", i.e., similar neighbors from a data set for every other object in the same data set.
    * Implementations could be either exact or approximate.
    *
    * @param dataPoints a row oriented matrix. Each row in the matrix represents an item in the data set. Items are
    *                   identified by their matrix index.
    * @param pred a prediction of vector pairs. Only collided pairs which suffices this  prediction will be returned.
    * @return a row oriented similarity matrix. Each row in the matrix represents an nearest neighbors, i.e., the
    *         indices corresponds the neighboring data row index and the matrix components hold the distance.
    */
  def apply(dataPoints: IndexedRowMatrix, pred: (Vector, Vector) => Boolean = {
    case (_, _) => true
  }): CoordinateMatrix
}

object SimilarityJoin {

  /**
    * Generates CosineDistanceJoin from the specified context.
    *
    * @param quantity the maximum number of data points to be joined
    * @param storageLevel the storage level when buckets persist
    * @param fraction expected size of the sample as a fraction of this RDD's size without replacement
    * @return a simhash join
    */
  def withCosineDistance(quantity: Int,
                         storageLevel: StorageLevel =
                           StorageLevel.MEMORY_AND_DISK,
                         fraction: Double = 1.0): SimilarityJoin =
    CosineDistanceJoin(quantity, storageLevel, fraction)

  /**
    * Generates JaccardDistanceJoin from the specified context.
    *
    * @param quantity the maximum number of data points to be joined
    * @param storageLevel the storage level when buckets persist
    * @param fraction expected size of the sample as a fraction of this RDD's size without replacement
    * @return a simhash join
    */
  def withJaccardDistance(quantity: Int,
                          storageLevel: StorageLevel =
                            StorageLevel.MEMORY_AND_DISK,
                          fraction: Double = 1.0): SimilarityJoin =
    JaccardDistanceJoin(quantity, storageLevel, fraction)

  /**
    * Generates ManhattanDistanceJoin from the specified context.
    *
    * @param quantity the maximum number of data points to be joined
    * @param storageLevel the storage level when buckets persist
    * @param fraction expected size of the sample as a fraction of this RDD's size without replacement
    * @return a simhash join
    */
  def withManhattanDistance(quantity: Int,
                            storageLevel: StorageLevel =
                              StorageLevel.MEMORY_AND_DISK,
                            fraction: Double = 1.0): SimilarityJoin =
    ManhattanDistanceJoin(quantity, storageLevel, fraction)

  /**
    * Generates EuclideanDistanceJoin from the specified context.
    *
    * @param quantity the maximum number of data points to be joined
    * @param storageLevel the storage level when buckets persist
    * @param fraction expected size of the sample as a fraction of this RDD's size without replacement
    * @return a simhash join
    */
  def withEuclideanDistance(quantity: Int,
                            storageLevel: StorageLevel =
                              StorageLevel.MEMORY_AND_DISK,
                            fraction: Double = 1.0): SimilarityJoin =
    EuclideanDistanceJoin(quantity, storageLevel, fraction)

  /**
    * Generates HammingDistanceJoin from the specified context.
    *
    * @param quantity the maximum number of data points to be joined
    * @param storageLevel the storage level when buckets persist
    * @param fraction expected size of the sample as a fraction of this RDD's size without replacement
    * @return a simhash join
    */
  def withHammingDistance(quantity: Int,
                          storageLevel: StorageLevel =
                            StorageLevel.MEMORY_AND_DISK,
                          fraction: Double = 1.0): SimilarityJoin =
    HammingDistanceJoin(quantity, storageLevel, fraction)
}

/**
  * Joins specified feature vectors with itself in accordance to the similarities. The resulting similarity matrix
  * is represented in the form of [[IndexedRowMatrix]].
  *
  * @author Shingo OKAWA
  */
trait ApproximateSimilarityJoin extends SimilarityJoin {

  /** Holds the associating [[Amplification]] implementation. */
  protected def amplification: Amplification

  /** The associating distance measure of this joiner. */
  override def distance: Distance = amplification.distance

  /** Holds the hash tables in the form of corresponding hash functions. */
  protected def hashTables: Seq[_ <: HashFunction[_ <: Digest[_]]]

  /**
    * Converts each data point into hash bucket entries.
    *
    * @param dataPoints a row oriented matrix. Each row in the matrix represents an item in the data set. Items are
    *                   identified by their matrix index.
    * @return a RDD of bucket entries.
    */
  private def generateBucketEntries(
      dataPoints: IndexedRowMatrix): RDD[_ <: BucketEntry[_]] =
    dataPoints.rows.flatMap { row =>
      hashTables.zipWithIndex.map {
        case (hash, table) =>
          hash(row.index, table, row.vector)
      }
    }

  /**
    * Caluculates the average selectivity of the data points in the data set.
    *
    * @param dataPoints a row oriented matrix. Each row in the matrix represents an item in the data set. Items are
    *                   identified by their matrix index.
    * @return the average selectivity of the data points.
    */
  def avgSelectivity(dataPoints: IndexedRowMatrix): Double = {
    val bucketEntries = generateBucketEntries(dataPoints)
    bucketEntries.persist(storageLevel)
    amplification.avgSelectivity(bucketEntries, dataPoints.rows.count())
  }

  /**
    * Finds the "nearest", i.e., similar neighbors from a data set for every other object in the same data set.
    * Implementations could be either exact or approximate.
    *
    * @param dataPoints a row oriented matrix. Each row in the matrix represents an item in the data set. Items are
    *                   identified by their matrix index.
    * @param pred a prediction of vector pairs. Only collided pairs which suffices this  prediction will be returned.
    * @return a row oriented similarity matrix. Each row in the matrix represents an nearest neighbors, i.e., the
    *         indices corresponds the neighboring data row index and the matrix components hold the distance.
    */
  override def apply(dataPoints: IndexedRowMatrix,
                     pred: (Vector, Vector) => Boolean = {
                       case (_, _) => true
                     }): CoordinateMatrix = {
    val bucketEntries = generateBucketEntries(dataPoints)
    bucketEntries.persist(storageLevel)
    new CoordinateMatrix(amplification(bucketEntries, quantity, pred),
                         dataPoints.numRows,
                         dataPoints.numRows)
  }
}

object ApproximateSimilarityJoin {

  /**
    * Generates SimHashJoin from the specified context.
    *
    * @param quantity the maximum number of data points to be joined
    * @param numFeatures the dimention of the feature vector to be hashed
    * @param signatureLength the hash length
    * @param numTables the number of hash functions
    * @param random a random number generator
    * @param storageLevel the storage level when buckets persist
    * @return a simhash join
    */
  def withCosineDistance(quantity: Int,
                         numFeatures: Int,
                         signatureLength: Int,
                         numTables: Int,
                         random: JRandom = new JRandom,
                         storageLevel: StorageLevel =
                           StorageLevel.MEMORY_AND_DISK)
    : ApproximateSimilarityJoin =
    SimHashJoin(quantity,
                numFeatures,
                signatureLength,
                numTables,
                random,
                storageLevel)

  /**
    * Generates MinHashJoin from the specified context.
    *
    * @param quantity the maximum number of data points to be joined
    * @param numFeatures the dimention of the feature vector to be hashed
    * @param signatureLength the hash length
    * @param numTables the number of hash functions
    * @param numBands the number of bands
    * @param prime the prime number used to generate random permutation
    * @param random a random number generator
    * @param storageLevel the storage level when buckets persist
    * @return a simhash join
    */
  def withJaccardDistance(quantity: Int,
                          numFeatures: Int,
                          signatureLength: Int,
                          numTables: Int,
                          numBands: Int,
                          prime: Int,
                          random: JRandom = new JRandom,
                          storageLevel: StorageLevel =
                            StorageLevel.MEMORY_AND_DISK)
    : ApproximateSimilarityJoin =
    MinHashJoin(quantity,
                numFeatures,
                signatureLength,
                numTables,
                numBands,
                prime,
                random,
                storageLevel)

  /**
    * Generates PStableL1Join from the specified context.
    *
    * @param quantity the maximum number of data points to be joined
    * @param numFeatures the dimention of the feature vector to be hashed
    * @param signatureLength the hash length
    * @param numTables the number of hash functions
    * @param bucketWidth the width to be used when truncating hash values to integers
    * @param random a random number generator
    * @param storageLevel the storage level when buckets persist
    * @return a simhash join
    */
  def withManhattanDistance(quantity: Int,
                            numFeatures: Int,
                            signatureLength: Int,
                            numTables: Int,
                            bucketWidth: Double,
                            random: JRandom = new JRandom,
                            storageLevel: StorageLevel =
                              StorageLevel.MEMORY_AND_DISK)
    : ApproximateSimilarityJoin =
    PStableL1Join(quantity,
                  numFeatures,
                  signatureLength,
                  numTables,
                  bucketWidth,
                  random,
                  storageLevel)

  /**
    * Generates PStableL2Join from the specified context.
    *
    * @param quantity the maximum number of data points to be joined
    * @param numFeatures the dimention of the feature vector to be hashed
    * @param signatureLength the hash length
    * @param numTables the number of hash functions
    * @param bucketWidth the width to be used when truncating hash values to integers
    * @param random a random number generator
    * @param storageLevel the storage level when buckets persist
    * @return a simhash join
    */
  def withEuclideanDistance(quantity: Int,
                            numFeatures: Int,
                            signatureLength: Int,
                            numTables: Int,
                            bucketWidth: Double,
                            random: JRandom = new JRandom,
                            storageLevel: StorageLevel =
                              StorageLevel.MEMORY_AND_DISK)
    : ApproximateSimilarityJoin =
    PStableL2Join(quantity,
                  numFeatures,
                  signatureLength,
                  numTables,
                  bucketWidth,
                  random,
                  storageLevel)

  /**
    * Generates BitSamplingJoin from the specified context.
    *
    * @param quantity the maximum number of data points to be joined
    * @param numFeatures the dimention of the feature vector to be hashed
    * @param signatureLength the hash length
    * @param numTables the number of hash functions
    * @param random a random number generator
    * @param storageLevel the storage level when buckets persist
    * @return a simhash join
    */
  def withHammingDistance(quantity: Int,
                          numFeatures: Int,
                          signatureLength: Int,
                          numTables: Int,
                          random: JRandom = new JRandom,
                          storageLevel: StorageLevel =
                            StorageLevel.MEMORY_AND_DISK)
    : ApproximateSimilarityJoin =
    BitSamplingJoin(quantity,
                    numFeatures,
                    signatureLength,
                    numTables,
                    random,
                    storageLevel)
}
