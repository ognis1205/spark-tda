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

import scala.collection.immutable.BitSet
import org.apache.spark.mllib.linalg.Vector

/**
  * Digest represents hash values which will be used LSH hashing.
  *
  * @author Shingo OKAWA
  */
private[distributed] sealed trait Digest[+T] extends Any {

  /** Holds hash value. */
  val value: T
}

/**
  * BucketEntry contains a row index of [[IndexedRowMatrix]], a digest and a hash table number.
  *
  * @author Shingo OKAWA
  */
private[distributed] sealed trait BucketEntry[+D <: Digest[_]]
    extends Serializable {

  /** Holds a row index. */
  val index: Long

  /** Holds hash table number. */
  val table: Int

  /** Holds digest. */
  val digest: D

  /** Holds original vector. */
  val vector: Vector

  /** Returns elements of digest. */
  def signature: Seq[Int]
}

/**
  * Digest with [[BitSet]] values.
  *
  * @author Shingo OKAWA
  */
private[impl] final case class BitSetDigest(value: BitSet)
    extends AnyVal
    with Digest[BitSet]

/**
  * Digest with [[Seq]][Int] values.
  *
  * @author Shingo OKAWA
  */
private[impl] final case class IntSeqDigest(value: Seq[Int])
    extends AnyVal
    with Digest[Seq[Int]]

/**
  * BucketEntry of [[BitSetDigest]].
  *
  * @author Shingo OKAWA
  */
private[impl] final case class BitSetBucketEntry(
    index: Long,
    table: Int,
    digest: BitSetDigest,
    vector: Vector
) extends BucketEntry[BitSetDigest] {

  /** Returns elements of digest. */
  override def signature: Seq[Int] = digest.value.toSeq
}

/**
  * BucketEntry of [[IntSeqDigest]].
  *
  * @author Shingo OKAWA
  */
private[impl] final case class IntSeqBucketEntry(
    index: Long,
    table: Int,
    digest: IntSeqDigest,
    vector: Vector
) extends BucketEntry[IntSeqDigest] {

  /** Returns elements of digest. */
  override def signature: Seq[Int] = digest.value
}
