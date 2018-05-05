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

import scala.util.hashing.MurmurHash3
import org.apache.spark.mllib.linalg.{Distance, SparseVector, Vector}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, MatrixEntry}
import org.apache.spark.mllib.rdd.MLPairRDDFunctions._
import org.apache.spark.rdd.RDD

/**
  * Amplification represents collision strategies for LSH.
  *
  * @author Shingo OKAWA
  */
private[distributed] sealed trait Amplification extends Serializable {

  /** Each buckets represented by the [[Iterable]][[[IndexedRow]]]. */
  protected type Bucket = Iterable[IndexedRow]

  /** The order to be used for taking the nearest neighbors of the specified vector. */
  private val ordering = Ordering[Double].on[(Long, Double)](_._2).reverse

  /** The associating distance measure of this amplification. */
  val distance: Distance

  /**
    * Generates a hash table from the specified bucket entries.
    *
    * @param bucketEntries the bucket entries to be bucketized
    * @return a hash table in the form of [[RDD]][[[Bucket]]]
    */
  protected def bucketize(bucketEntries: RDD[_ <: BucketEntry[_]]): RDD[Bucket]

  /**
    * Caluculates the average selectivity of the data points in the data set.
    *
    * @param bucketEntries the bucket entries to be bucketized
    * @param numPoints the number of data points
    * @return an average selectivity of the data points
    */
  def avgSelectivity(bucketEntries: RDD[_ <: BucketEntry[_]],
                     numPoints: Long): Double =
    bucketize(bucketEntries)
      .flatMap { bucket =>
        for {
          l <- bucket.iterator
          r <- bucket.iterator
        } yield (l.index, r.index)
      }
      .distinct()
      .countByKey()
      .values
      .map(_.toDouble / numPoints)
      .reduce(_ + _) / numPoints

  /**
    * Generates all collided pairs and its distance in the form of [[MatrixEntry]].
    *
    * @param bucketEntries the bucket entries to be bucketized
    * @param quantity at most number of neighbors will be returned
    * @param pred a prediction of vector pairs. Only collided pairs which suffices this  prediction will be returned.
    * @return a similarity matrix in the form of [[RDD]][[[MatrixEntry]]].
    */
  def apply(bucketEntries: RDD[_ <: BucketEntry[_]],
            quantity: Int,
            pred: (Vector, Vector) => Boolean = { case (_, _) => true })
    : RDD[MatrixEntry] =
    bucketize(bucketEntries)
      .flatMap { bucket =>
        for {
          l <- bucket.iterator
          r <- bucket.iterator
          if l.index < r.index && pred(l.vector, r.vector)
        } yield ((l.index, r.index), distance(l.vector, r.vector))
      }
      .reduceByKey((l, r) => l)
      .flatMap {
        case ((l, r), d) => Array((l, (r, d)), (r, (l, d)))
      }
      .topByKey(quantity)(ordering)
      .flatMap {
        case (i, arr) =>
          arr.map {
            case (j, d) =>
              new MatrixEntry(i, j, d)
          }
      }
}

/**
  * OR-constructrion for LSH.
  *
  * @author Shingo OKAWA
  */
private[impl] final class ORConstruction private[impl] (
    override val distance: Distance)
    extends Amplification {

  /**
    * Generates a hash table from the specified bucket entries.
    *
    * @param bucketEntries the bucket entries to be bucketized
    * @return a hash table in the form of [[RDD]][[[Bucket]]]
    */
  override def bucketize(
      bucketEntries: RDD[_ <: BucketEntry[_]]): RDD[Bucket] =
    bucketEntries
      .map(entry => {
        val key = (entry.table, MurmurHash3.seqHash(entry.signature))
          .asInstanceOf[Product]
        (key, new IndexedRow(entry.index, entry.vector))
      })
      .groupByKey(bucketEntries.getNumPartitions)
      .values
}

private[impl] object ORConstruction {

  /**
    * Generates ORConstruction from the assigned distance measure.
    *
    * @param distance the distance measure to be caluculated
    * @return an OR-construction amplifier
    */
  def apply(distance: Distance): ORConstruction =
    new ORConstruction(distance)
}

/**
  * Band OR-constructrion for LSH.
  *
  * @author Shingo OKAWA
  */
private[impl] final class BandORConstruction private[impl] (
    override val distance: Distance,
    val numBands: Int)
    extends Amplification {

  /**
    * Generates a hash table from the specified bucket entries.
    *
    * @param bucketEntries the bucket entries to be bucketized
    * @return a hash table in the form of [[RDD]][[[Bucket]]]
    */
  override def bucketize(
      bucketEntries: RDD[_ <: BucketEntry[_]]): RDD[Bucket] =
    bucketEntries
      .flatMap(entry => {
        val signature = entry.signature
        signature.grouped(signature.size / numBands).zipWithIndex.map {
          case (band, bandIndex) => {
            val key = (entry.table, bandIndex, MurmurHash3.seqHash(band))
              .asInstanceOf[Product]
            (key, new IndexedRow(entry.index, entry.vector))
          }
        }
      })
      .groupByKey(bucketEntries.getNumPartitions)
      .values
}

private[impl] object BandORConstruction {

  /**
    * Generates BandORConstruction from the assigned distance measure.
    *
    * @param distance the distance measure to be caluculated
    * @param numBands the number of bands
    * @return an OR-construction amplifier
    */
  def apply(distance: Distance, numBands: Int): BandORConstruction =
    new BandORConstruction(distance, numBands)
}
