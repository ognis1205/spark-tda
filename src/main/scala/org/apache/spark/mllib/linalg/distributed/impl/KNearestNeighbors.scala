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

import org.apache.spark.mllib.linalg.{
  Vector,
  Distance,
  CosineDistance,
  JaccardDistance,
  ManhattanDistance,
  EuclideanDistance,
  HammingDistance
}
import org.apache.spark.mllib.linalg.distributed.SimilarityJoin
import org.apache.spark.mllib.linalg.distributed.{
  MatrixEntry,
  IndexedRowMatrix,
  CoordinateMatrix
}
import org.apache.spark.mllib.rdd.MLPairRDDFunctions._
import org.apache.spark.storage.StorageLevel

/**
  * Joins specified feature vectors with itself in accordance to specified the similarities. The resulting similarity
  * matrix is represented in the form of [[IndexedRowMatrix]].
  *
  * @author Shingo OKAWA
  */
private[distributed] final class KNearestNeighbor private[impl] (
    override val distance: Distance,
    override val quantity: Int,
    override val storageLevel: StorageLevel,
    val fraction: Double
) extends SimilarityJoin {

  /** The order to be used for taking the nearest neighbors of the specified vector. */
  private val ordering = Ordering[Double].on[(Long, Double)](_._2).reverse

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
    val samples = dataPoints.rows.sample(false, fraction)
    if (fraction != 1.0) samples.persist(storageLevel)
    new CoordinateMatrix(
      (samples cartesian dataPoints.rows)
        .flatMap {
          case (lhs, rhs) =>
            if (lhs.index < rhs.index && pred(lhs.vector, rhs.vector))
              Some(((lhs.index, rhs.index), distance(lhs.vector, rhs.vector)))
            else None
        }
        .reduceByKey((l, r) => l)
        .flatMap { case ((i, j), d) => Array((i, (j, d)), (j, (i, d))) }
        .topByKey(quantity)(ordering)
        .flatMap {
          case (i, arr) =>
            arr.map { case (j, d) => new MatrixEntry(i, j, d) }
        }
    )
  }
}

private[distributed] object CosineDistanceJoin {

  /**
    * Generates cosine distance join from the specified context.
    *
    * @param quantity the maximum number of data points to be joined
    * @param storageLevel the storage level when buckets persist
    * @param fraction expected size of the sample as a fraction of this RDD's size without replacement
    * @return a simhash join
    */
  def apply(quantity: Int,
            storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
            fraction: Double = 1.0): KNearestNeighbor = {
    require(fraction >= 0.0 && fraction <= 1.0,
            "fraction must be within [0.0, 1.0].")
    new KNearestNeighbor(CosineDistance, quantity, storageLevel, fraction)
  }
}

private[distributed] object JaccardDistanceJoin {

  /**
    * Generates jaccard distance join from the specified context.
    *
    * @param quantity the maximum number of data points to be joined
    * @param storageLevel the storage level when buckets persist
    * @param fraction expected size of the sample as a fraction of this RDD's size without replacement
    * @return a simhash join
    */
  def apply(quantity: Int,
            storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
            fraction: Double = 1.0): KNearestNeighbor = {
    require(fraction >= 0.0 && fraction <= 1.0,
            "fraction must be within [0.0, 1.0].")
    new KNearestNeighbor(JaccardDistance, quantity, storageLevel, fraction)
  }
}

private[distributed] object ManhattanDistanceJoin {

  /**
    * Generates manhattan distance join from the specified context.
    *
    * @param quantity the maximum number of data points to be joined
    * @param storageLevel the storage level when buckets persist
    * @param fraction expected size of the sample as a fraction of this RDD's size without replacement
    * @return a simhash join
    */
  def apply(quantity: Int,
            storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
            fraction: Double = 1.0): KNearestNeighbor = {
    require(fraction >= 0.0 && fraction <= 1.0,
            "fraction must be within [0.0, 1.0].")
    new KNearestNeighbor(ManhattanDistance, quantity, storageLevel, fraction)
  }
}

private[distributed] object EuclideanDistanceJoin {

  /**
    * Generates euclidean distance join from the specified context.
    *
    * @param quantity the maximum number of data points to be joined
    * @param storageLevel the storage level when buckets persist
    * @param fraction expected size of the sample as a fraction of this RDD's size without replacement
    * @return a simhash join
    */
  def apply(quantity: Int,
            storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
            fraction: Double = 1.0): KNearestNeighbor = {
    require(fraction >= 0.0 && fraction <= 1.0,
            "fraction must be within [0.0, 1.0].")
    new KNearestNeighbor(EuclideanDistance, quantity, storageLevel, fraction)
  }
}

private[distributed] object HammingDistanceJoin {

  /**
    * Generates hamming distance join from the specified context.
    *
    * @param quantity the maximum number of data points to be joined
    * @param storageLevel the storage level when buckets persist
    * @param fraction expected size of the sample as a fraction of this RDD's size without replacement
    * @return a simhash join
    */
  def apply(quantity: Int,
            storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
            fraction: Double = 1.0): KNearestNeighbor = {
    require(fraction >= 0.0 && fraction <= 1.0,
            "fraction must be within [0.0, 1.0].")
    new KNearestNeighbor(HammingDistance, quantity, storageLevel, fraction)
  }
}
