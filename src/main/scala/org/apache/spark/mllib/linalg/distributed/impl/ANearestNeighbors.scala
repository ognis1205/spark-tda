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
import org.apache.spark.mllib.linalg.{
  CosineDistance,
  JaccardDistance,
  ManhattanDistance,
  EuclideanDistance,
  HammingDistance
}
import org.apache.spark.mllib.linalg.distributed.ApproximateSimilarityJoin
import org.apache.spark.storage.StorageLevel

/**
  * Joins specified feature vectors with itself in accordance to the specified similarities. The resulting
  * similarity matrix is represented in the form of [[IndexedRowMatrix]].
  *
  * @author Shingo OKAWA
  */
private[distributed] final class ANearestNeighbor private[impl] (
    override val amplification: Amplification,
    override val hashTables: Seq[_ <: HashFunction[_ <: Digest[_]]],
    override val quantity: Int,
    override val storageLevel: StorageLevel
) extends ApproximateSimilarityJoin

private[distributed] object SimHashJoin {

  /**
    * Generates simhash join from the specified context.
    *
    * @param quantity the maximum number of data points to be joined
    * @param numFeatures the dimention of the feature vector to be hashed
    * @param signatureLength the hash length
    * @param numTables the number of hash functions
    * @param random a random number generator
    * @param storageLevel the storage level when buckets persist
    * @return a simhash join
    */
  def apply(quantity: Int,
            numFeatures: Int,
            signatureLength: Int,
            numTables: Int,
            random: JRandom = new JRandom,
            storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK)
    : ANearestNeighbor =
    new ANearestNeighbor(
      ORConstruction(CosineDistance),
      (0 until numTables)
        .map(_ => SimHash(numFeatures, signatureLength, random))
        .toSeq,
      quantity,
      storageLevel
    )
}

private[distributed] object MinHashJoin {

  /**
    * Generates minhash join from the specified context.
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
  def apply(quantity: Int,
            numFeatures: Int,
            signatureLength: Int,
            numTables: Int,
            numBands: Int,
            prime: Int,
            random: JRandom = new JRandom,
            storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK)
    : ANearestNeighbor = {
    require(prime > 0,
            "the seed number to generate random permutation must be positive")
    require(
      numBands > 0 && signatureLength % numBands == 0,
      "the number of bands must be positive and the signature length must be devidable by the number of bands")
    new ANearestNeighbor(
      BandORConstruction(JaccardDistance, numBands),
      (0 until numTables)
        .map(_ => MinHash(numFeatures, signatureLength, prime, random))
        .toSeq,
      quantity,
      storageLevel
    )
  }
}

private[distributed] object PStableL1Join {

  /**
    * Generates p-stable L1 join from the specified context.
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
  def apply(quantity: Int,
            numFeatures: Int,
            signatureLength: Int,
            numTables: Int,
            bucketWidth: Double,
            random: JRandom = new JRandom,
            storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK)
    : ANearestNeighbor = {
    require(bucketWidth > 0.0, "bucket width must be positive")
    new ANearestNeighbor(
      ORConstruction(ManhattanDistance),
      (0 until numTables)
        .map(_ =>
          PStable.L1(numFeatures, signatureLength, bucketWidth, random))
        .toSeq,
      quantity,
      storageLevel
    )
  }
}

private[distributed] object PStableL2Join {

  /**
    * Generates p-stable L2 join from the specified context.
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
  def apply(quantity: Int,
            numFeatures: Int,
            signatureLength: Int,
            numTables: Int,
            bucketWidth: Double,
            random: JRandom = new JRandom,
            storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK)
    : ANearestNeighbor = {
    require(bucketWidth > 0.0, "bucket width must be positive")
    new ANearestNeighbor(
      ORConstruction(EuclideanDistance),
      (0 until numTables)
        .map(_ =>
          PStable.L2(numFeatures, signatureLength, bucketWidth, random))
        .toSeq,
      quantity,
      storageLevel
    )
  }
}

private[distributed] object BitSamplingJoin {

  /**
    * Generates bit sampling join from the specified context.
    *
    * @param quantity the maximum number of data points to be joined
    * @param numFeatures the dimention of the feature vector to be hashed
    * @param signatureLength the hash length
    * @param numTables the number of hash functions
    * @param random a random number generator
    * @param storageLevel the storage level when buckets persist
    * @return a simhash join
    */
  def apply(quantity: Int,
            numFeatures: Int,
            signatureLength: Int,
            numTables: Int,
            random: JRandom = new JRandom,
            storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK)
    : ANearestNeighbor =
    new ANearestNeighbor(
      ORConstruction(HammingDistance),
      (0 until numTables)
        .map(_ => BitSampling(numFeatures, signatureLength, random))
        .toSeq,
      quantity,
      storageLevel
    )
}
