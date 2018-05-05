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

import scala.collection.mutable.PriorityQueue
import scala.util.Random
import breeze.linalg.DenseVector
import breeze.stats.meanAndVariance
import org.apache.spark.ml.linalg.{Vector, Vectors, Distance}
import org.apache.spark.util.random.XORShiftRandom

/**
  * A `Tree` is used to index data points to find approximated k-nearest neighbors.
  *
  * @author Shingo OKAWA
  */
private[ml] sealed abstract class Tree
    extends Iterable[VectorWithId]
    with Serializable {
  import Tree._

  /** Left tree. */
  val left: Tree

  /** Right tree. */
  val right: Tree

  /** Holds the cardinality of the underlying vector space. */
  val cardinality: Int

  /** Holds the number of descendant leaves. */
  val numLeaves: Int

  /** Queries the k-NN of the specified vector. */
  def query(vectorWithId: VectorWithId, k: Int = 1): KNN =
    collect(KNN(vectorWithId, k))

  /** Refines k-NN candidatse using data resides in this `Tree`. */
  private[knn] def collect(knn: KNN): KNN
}

private[ml] object Tree {

  /** A `KNN` is used to maintain search progress for single query vector. */
  private[ml] final class KNN private (
      val query: VectorWithId,
      val k: Int,
      val candidates: PriorityQueue[(VectorWithId, Double)] =
        PriorityQueue.empty[(VectorWithId, Double)] { Ordering.by(_._2) }
  ) extends Iterable[(VectorWithId, Double)]
      with Serializable {

    /** Returns the `Iterator` of k-NN. */
    override def iterator: Iterator[(VectorWithId, Double)] =
      candidates.iterator

    /** Caluculates the radius of this k-NN. */
    def radius: Double = if (candidates.isEmpty) 0.0 else candidates.head._2

    /** Returns `true` if the k-NN has k members. */
    def isFull: Boolean = candidates.size >= k

    /** Returns `true` if the k-NN is not fulfilled. */
    def isNotFull: Boolean = !isFull

    /** Enqueues the specified vector without requiring decresing the k-NN's radius. */
    def enqueue(vectorWithId: VectorWithId, distance: Double): Unit = {
      while (isFull) candidates.dequeue()
      candidates.enqueue((vectorWithId, distance))
    }

    /** Inserts the specified vector if the k-NN radius shrinks. */
    def insert(vectorWithId: VectorWithId, distance: Double): Unit =
      if (isNotFull || distance < radius) enqueue(vectorWithId, distance)
  }

  private[ml] object KNN {

    /** Instanciates `KNN` from the given context. */
    def apply(query: VectorWithId, k: Int): KNN = new KNN(query, k)
  }
}

/**
  *  An `Sentinel` represents sentinel `Tree`.
  *
  * @author Shingo OKAWA
  */
private[ml] final case object Sentinel extends Tree {
  import Tree._

  /** Left tree. */
  override val left: Tree = this

  /** Right tree. */
  override val right: Tree = this

  /** Holds the cardinality of the underlying vector space. */
  override val cardinality: Int = 0

  /** Holds the number of descendants. */
  override val numLeaves: Int = 0

  /** Refines k-NN candidatse using data resides in this `MetricTree`. */
  override def collect(knn: KNN): KNN = knn

  /** Returns the `Iterator` of the tree. */
  override def iterator: Iterator[VectorWithId] = Iterator.empty
}

/**
  * A `MetricTree` represents a binary metric tree node which orders data points in according to the specified
  * measure.
  *
  * @author Shingo OKAWA
  */
private[ml] sealed abstract class MetricTree extends Tree {

  /** Holds the measure of the underlying vector space. */
  val measure: Distance
}

/**
  *  A `Leaf` represents leaf `Tree`.
  *
  * @author Shingo OKAWA
  */
private[ml] final case class Leaf(
    data: IndexedSeq[VectorWithId],
    override val measure: Distance
) extends MetricTree {
  import Tree._

  /** Left tree. */
  override val left: Tree = Sentinel

  /** Right tree. */
  override val right: Tree = Sentinel

  /** Holds the cardinality of the underlying vector space. */
  override val cardinality: Int = data.size

  /** Holds the number of descendants. */
  override val numLeaves: Int = 1

  /** Refines k-NN candidatse using data resides in this `VPTree`. */
  override def collect(knn: KNN): KNN = {
    val sorted = data
      .map { v =>
        (v, measure(knn.query.vector, v.vector))
      }
      .sortBy(_._2)
    for ((v, d) <- sorted if knn.isNotFull || d < knn.radius) knn.enqueue(v, d)
    knn
  }

  /** Returns the `Iterator` of the tree. */
  override def iterator: Iterator[VectorWithId] = data.iterator
}

/**
  * A `VPTree` represents a binary metric tree node and keeps track of the vantage point vector which closely
  * approximate the center of all vectors within the node.
  *
  * @author Shingo OKAWA
  */
private[ml] sealed abstract class VPTree extends MetricTree {
  import Tree._
  import VPTree._

  /** Holds the vantage point vector of the underlying vector space. */
  val vantagePoint: VantagePoint

  /** Computes the query cost defined as distance(vabtagePoint.vector, query.vector). */
  private[knn] def distance(query: VectorWithId): Double =
    measure(vantagePoint.vector, query.vector)

  /** Computes the query cost defined as distance(vantagePoint.vector, knn.query.vector). */
  private[knn] def distance(knn: KNN): Double = distance(knn.query)
}

private[ml] object VPTree {

  /** Represents vantage point. */
  case class VantagePoint(override val id: Long,
                          override val vector: Vector,
                          val median: Double)
      extends VectorWithId

  private[knn] object VantagePoint {

    /** Instanciates `VantagePoint` from the given context. */
    private[knn] def apply(data: IndexedSeq[VectorWithId],
                           numCandidates: Int,
                           numSamples: Int,
                           measure: Distance,
                           seed: Long = 0L): VantagePoint = {
      val random = new Random(seed)
      val candidates =
        if (data.size > numCandidates) random.shuffle(data).take(numCandidates)
        else data
      val samples =
        if (data.size > numSamples) random.shuffle(data).take(numSamples)
        else data
      val vantagePoint = candidates
        .map { candidate =>
          (candidate, DenseVector(samples.map { sample =>
            measure(candidate.vector, sample.vector)
          }: _*))
        }
        .map {
          case (candidate, distances) =>
            (candidate, meanAndVariance(distances).variance)
        }
        .maxBy { case (_, variance) => variance }
        ._1
      val distances = data
        .map { vectorWithId =>
          measure(vantagePoint.vector, vectorWithId.vector)
        }
        .sortWith(_ < _)
      val median =
        if (distances.size % 2 == 1) distances(distances.size / 2)
        else {
          val (smaller, bigger) = distances.splitAt(distances.size / 2)
          (smaller.last + bigger.head) / 2.0
        }
      new VantagePoint(vantagePoint.id, vantagePoint.vector, median)
    }
  }

  /**
    * Builds a `VPTree` that faciliates k-NN search.
    *
    * @param data vectors that contains all training data
    * @param numCandidates the number of candidates of vantage points to be chosen
    * @param numSamples the number of samples to be chosen for estimating the variance of distance
    * @param leafSize the number of data vectors which resides in a single leaf
    * @param seed random number generator seed used in selecting vantage piont
    * @return `Tree` can be used for k-NN search
    */
  def apply(data: IndexedSeq[VectorWithId],
            measure: Distance,
            numCandidates: Int,
            numSamples: Int,
            leafSize: Int = 1,
            seed: Long = 0L): Tree = {
    val size = data.size
    val rand = new XORShiftRandom(seed)
    if (size == 0) {
      Sentinel
    } else if (size <= leafSize) {
      Leaf(data, measure)
    } else {
      val vantagePoint =
        VantagePoint(data, numCandidates, numSamples, measure, rand.nextLong())
      if (vantagePoint.median == 0.0) {
        Leaf(data, measure)
      } else {
        val (left, right) = data
          .partition { vectorWithId =>
            measure(vantagePoint.vector, vectorWithId.vector) < vantagePoint.median
          }
        Branch(
          VPTree(left,
                 measure,
                 numCandidates,
                 numSamples,
                 leafSize,
                 rand.nextLong()),
          VPTree(right,
                 measure,
                 numCandidates,
                 numSamples,
                 leafSize,
                 rand.nextLong()),
          measure,
          vantagePoint
        )
      }
    }
  }
}

/**
  *  A `Branch` represents branch `VPTree`.
  *
  * @author Shingo OKAWA
  */
private[ml] final case class Branch(
    override val left: Tree,
    override val right: Tree,
    override val measure: Distance,
    override val vantagePoint: VPTree.VantagePoint
) extends VPTree {
  import Tree._
  import VPTree._

  /** Holds the cardinality of the underlying vector space. */
  override val cardinality: Int = left.cardinality + right.cardinality

  /** Holds the number of descendant leaves. */
  override val numLeaves: Int = left.numLeaves + right.numLeaves

  /** Refines k-NN candidatse using data resides in this `VPTree`. */
  override def collect(knn: KNN): KNN = {
    val d = distance(knn)
    // TODO: The following algorithm can be mode effective by implementing more sophisticated pruning.
    if (d < vantagePoint.median) {
      left.collect(knn)
      if (knn.isNotFull || d + knn.radius >= vantagePoint.median)
        right.collect(knn)
    } else {
      right.collect(knn)
      if (knn.isNotFull || d - knn.radius < vantagePoint.median)
        left.collect(knn)
    }
    knn
  }

  /** Returns the `Iterator` of the tree. */
  override def iterator: Iterator[VectorWithId] =
    left.iterator ++ right.iterator
}
