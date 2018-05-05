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
import scala.collection.mutable.ArrayBuilder
import scala.collection.immutable.BitSet
import org.apache.spark.mllib.linalg.{
  Vector,
  Projection,
  GaussianRandomProjection,
  CauchyRandomProjection
}

/**
  * HashFunction represents the hasher for LSH.
  *
  * @author Shingo OKAWA
  */
private[distributed] sealed trait HashFunction[+D <: Digest[_]]
    extends Serializable {

  /**
    * Caluculates the bucket entry for the specified context.
    *
    * @param index the row index of the vector
    * @param table the hash table number
    * @param vector the vector to be hashed
    * @return a bucket entry
    */
  def apply(index: Long, table: Int, vector: Vector): BucketEntry[D]
}

/**
  * HashFunction for SimHash. This implementation is based on the following report:
  * [[https://www.cs.princeton.edu/courses/archive/spr04/cos598B/bib/CharikarEstim.pdf
  *   Similarity Estimation Techniques from Rounding Algorithms]]
  *
  * @see http://ilpubs.stanford.edu:8090/1077/3/p535-salihoglu.pdf
  * @author Shingo OKAWA
  */
private[impl] final class SimHash private[impl] (
    projection: Projection
) extends HashFunction[BitSetDigest] {

  /**
    * Caluculates the bucket entry for the specified context.
    *
    * @param index the row index of the vector
    * @param table the hash table number
    * @param vector the vector to be hashed
    * @return a bucket entry
    */
  override def apply(index: Long,
                     table: Int,
                     vector: Vector): BucketEntry[BitSetDigest] = {
    val projected = projection(vector)
    val bits = ArrayBuilder.make[Int]
    projected.foreachActive {
      case (i, v) =>
        if (v > 0.0) bits += i
    }
    BitSetBucketEntry(index,
                      table,
                      BitSetDigest(BitSet(bits.result(): _*)),
                      vector)
  }
}

private[impl] object SimHash {

  /**
    * Generates SimHash function from the given context.
    *
    * @param numFeatures the dimention of the feature vector to be hashed
    * @param signatureLength the hash length
    * @param random a random number generator
    * @return a simhash function
    */
  def apply(numFeatures: Int,
            signatureLength: Int,
            random: JRandom = new JRandom): SimHash =
    new SimHash(GaussianRandomProjection(numFeatures, signatureLength, random))
}

/**
  * HashFunction for MinHash.
  *
  * @see https://en.wikipedia.org/wiki/MinHash
  * @author Shingo OKAWA
  */
private[impl] final class MinHash private[impl] (
    permutations: Seq[RandomPermutation]
) extends HashFunction[IntSeqDigest] {

  /**
    * Caluculates the bucket entry for the specified context.
    *
    * @param index the row index of the vector
    * @param table the hash table number
    * @param vector the vector to be hashed
    * @return a bucket entry
    */
  override def apply(index: Long,
                     table: Int,
                     vector: Vector): BucketEntry[IntSeqDigest] =
    IntSeqBucketEntry(
      index,
      table,
      IntSeqDigest(
        permutations.map(p => vector.toSparse.indices.map(p(_)).min)
      ),
      vector)
}

private[impl] object MinHash {

  /**
    * Generates MinHash function from the given context.
    *
    * @param numFeatures the dimention of the feature vector to be hashed
    * @param signatureLength the hash length
    * @param prime the prime number used to generate random permutation
    * @param random a random number generator
    * @return a simhash function
    */
  def apply(numFeatures: Int,
            signatureLength: Int,
            prime: Int,
            random: JRandom = new JRandom): MinHash = {
    require(isPrime(prime),
            "the seed number to generate random permutation should be prime")
    new MinHash(((0 until signatureLength).map(_ =>
      RandomPermutation(numFeatures, prime, random))))
  }

  /** Returns `true` if the number is prime. */
  private def isPrime(x: Int): Boolean =
    (x > 1) && !(2 to scala.math.sqrt(x).toInt).exists(y => x % y == 0)
}

/**
  * HashFunction for PStable. This implementation is based on the following report:
  * [[https://graphics.stanford.edu/courses/cs468-06-fall/Papers/12%20lsh04.pdf
  *   Locality-Sensitive Hashing Scheme Based on p-Stable Distributions]]
  *
  * @see https://graphics.stanford.edu/courses/cs468-06-fall/Papers/12%20lsh04.pdf
  * @author Shingo OKAWA
  */
private[impl] final class PStable private[impl] (
    projection: Projection,
    bs: Seq[Double],
    bucketWidth: Double
) extends HashFunction[IntSeqDigest] {

  /**
    * Caluculates the bucket entry for the specified context.
    *
    * @param index the row index of the vector
    * @param table the hash table number
    * @param vector the vector to be hashed
    * @return a bucket entry
    */
  override def apply(index: Long,
                     table: Int,
                     vector: Vector): BucketEntry[IntSeqDigest] = {
    val as = projection(vector)
    val sig = ArrayBuilder.make[Int]
    as.foreachActive {
      case (i, v) =>
        sig += math.floor((as(i) + bs(i)) / bucketWidth).toInt
    }
    IntSeqBucketEntry(index, table, IntSeqDigest(sig.result()), vector)
  }
}

private[impl] object PStable {

  /**
    * Generates PStable function for manhattan distance from the given context.
    *
    * @param numFeatures the dimention of the feature vector to be hashed
    * @param signatureLength the hash length
    * @param bucketWidth the width to use when truncating hash values to integers
    * @param random a random number generator
    * @return a simhash function
    */
  def L1(numFeatures: Int,
         signatureLength: Int,
         bucketWidth: Double,
         random: JRandom = new JRandom): PStable =
    new PStable(
      CauchyRandomProjection(numFeatures, signatureLength),
      offsetsFor(signatureLength, bucketWidth, random),
      bucketWidth
    )

  /**
    * Generates PStable function for euclidean distance from the given context.
    *
    * @param numFeatures the dimention of the feature vector to be hashed
    * @param signatureLength the hash length
    * @param bucketWidth the width to use when truncating hash values to integers
    * @param random a random number generator
    * @return a simhash function
    */
  def L2(numFeatures: Int,
         signatureLength: Int,
         bucketWidth: Double,
         random: JRandom = new JRandom): PStable =
    new PStable(
      GaussianRandomProjection(numFeatures, signatureLength, random),
      offsetsFor(signatureLength, bucketWidth, random),
      bucketWidth
    )

  /** Generats a set of offsets. */
  private def offsetsFor(signatureLength: Int,
                         bucketWidth: Double,
                         random: JRandom): Seq[Double] = {
    val offsets = ArrayBuilder.make[Double]
    for (i <- (0 until signatureLength)) {
      offsets += random.nextDouble() * bucketWidth
    }
    offsets.result()
  }
}

/**
  * HashFunction for BitSampling. This implementation is based on the following report:
  * [[http://www.vldb.org/conf/1999/P49.pdf
  *   Similarity Search in High Dimensions via Hashing]]
  *
  * @see http://www.vldb.org/conf/1999/P49.pdf
  * @author Shingo OKAWA
  */
private[impl] final class BitSampling private[impl] (
    mask: Seq[Int]
) extends HashFunction[BitSetDigest] {

  /**
    * Caluculates the bucket entry for the specified context.
    *
    * @param index the row index of the vector
    * @param table the hash table number
    * @param vector the vector to be hashed
    * @return a bucket entry
    */
  override def apply(index: Long,
                     table: Int,
                     vector: Vector): BucketEntry[BitSetDigest] =
    BitSetBucketEntry(
      index,
      table,
      BitSetDigest(BitSet((vector.toSparse.indices.intersect(mask)): _*)),
      vector)
}

private[impl] object BitSampling {

  /**
    * Generates BitSampling function from the given context.
    *
    * @param numFeatures the dimention of the feature vector to be hashed
    * @param signatureLength the hash length
    * @param prime the prime number used to generate random permutation
    * @param random a random number generator
    * @return a simhash function
    */
  def apply(numFeatures: Int,
            signatureLength: Int,
            random: JRandom = new JRandom): BitSampling =
    new BitSampling(Array.fill(signatureLength) {
      random.nextInt(numFeatures)
    })
}
