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
package org.apache.spark.mllib.linalg

import java.util.{Random => JRandom}
import scala.annotation.tailrec
import scala.util.Random
import breeze.stats.distributions.CauchyDistribution

/**
  * A projection based on Spark's random generation and multiplication of dense matrices.
  *
  * @author Shingo OKAWA
  */
sealed trait Projection extends (Vector => Vector) with Serializable {

  /** Returns the dimension of the domain space. */
  def srcDim: Int

  /** Returns the dimension of the codomain space. */
  def dstDim: Int

  /**
    * Projects a vector based on the predefined projection matrix.
    *
    * @param vector a vector to be projected
    * @return a projected vector
    */
  protected def project(vector: Vector): Vector

  /**
    * Caluculates a projected vector based on the predefined projection matrix with additional validations.
    *
    * @param vector a vector to be projected
    * @return a projected vector
    */
  def apply(vector: Vector): Vector = {
    require(
      vector.size == srcDim,
      "the size of vector must be equal to the dimension of domain of the projection")
    val projected = project(vector)
    require(
      projected.size == dstDim,
      "the size of resulting vector must be equal to the dimension fo codomain of the projection")
    projected
  }
}

/**
  * A gaussian random projection.
  *
  * @author Shingo OKAWA
  */
final class GaussianRandomProjection private (matrix: Matrix)
    extends Projection {

  /** Returns the dimension of the domain space. */
  override def srcDim: Int = matrix.numCols

  /** Returns the dimension of the codomain space. */
  override def dstDim: Int = matrix.numRows

  /**
    * Computes a projected vector based on the predefined projection matrix.
    *
    * @param vector a vector to be projected
    * @return a projected vector
    */
  override def project(vector: Vector): Vector = matrix multiply vector

  /** Returns the mean value of the underlying matrix components. */
  def mean: Double = matrix.toArray.sum / (srcDim * dstDim)

  /** Returns the standard deviation of the underlying matrix components. */
  def stddev: Double = {
    val mu = mean
    matrix.toArray.map(x => math.pow((x - mu), 2.0)).sum / (srcDim * dstDim)
  }
}

object GaussianRandomProjection {

  /**
    * Generate a gaussian rnadom projection consisting of i.i.d. gaussian random numbers.
    *
    * @param srcDim number of rows of the matrix
    * @param dstDim number of columns of the matrix
    * @param random a random number generator
    * @return a i.i.d gaussian random projecttion
    */
  def apply(srcDim: Int,
            dstDim: Int,
            random: JRandom): GaussianRandomProjection =
    new GaussianRandomProjection(DenseMatrix.randn(dstDim, srcDim, random))
}

/**
  * A cauchy random projection.
  *
  * @author Shingo OKAWA
  */
final class CauchyRandomProjection private (matrix: Matrix)
    extends Projection {

  /** Returns the dimension of the domain space. */
  override def srcDim: Int = matrix.numCols

  /** Returns the dimension of the codomain space. */
  override def dstDim: Int = matrix.numRows

  /**
    * Computes a projected vector based on the predefined projection matrix.
    *
    * @param vector a vector to be projected
    * @return a projected vector
    */
  override def project(vector: Vector): Vector = matrix multiply vector

  /** Returns the k-th median value of the array. */
  @tailrec
  private def findKMedian(array: Array[Double], k: Int): Double = {
    import scala.language.postfixOps
    val pivot = array(Random.nextInt(array.size))
    val (smaller, bigger) = array partition (pivot >)
    if (smaller.size == k) pivot
    else if (smaller.isEmpty) {
      val (smaller, bigger) = array partition (pivot ==)
      if (smaller.size > k) pivot
      else findKMedian(bigger, k - smaller.size)
    } else if (smaller.size < k) findKMedian(bigger, k - smaller.size)
    else findKMedian(smaller, k)
  }

  /** Returns the median value of the array. */
  private def findMedian(array: Array[Double]): Double =
    findKMedian(array, (array.size - 1) / 2)

  /** Returns the median value of the underlying matrix components. */
  def median: Double = findMedian(matrix.toArray)
}

private[linalg] object CauchyRandomProjection {

  /**
    * Generate a cauchy random projection consisting of i.i.d. cauchy random numbers.
    *
    * @param srcDim number of rows of the matrix
    * @param dstDim number of columns of the matrix
    * @return a i.i.d gaussian random projecttion
    */
  def apply(srcDim: Int, dstDim: Int): CauchyRandomProjection =
    new CauchyRandomProjection(
      new DenseMatrix(dstDim,
                      srcDim,
                      new CauchyDistribution(0, 1).drawMany(dstDim * srcDim)))
}
