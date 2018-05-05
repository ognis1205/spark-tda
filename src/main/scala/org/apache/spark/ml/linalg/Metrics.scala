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
package org.apache.spark.ml.linalg

import breeze.linalg.norm
import org.apache.spark.mllib.util.MLUtils

/**
  * Distance represents the distance measurement between two vectors.
  *
  * @author Shingo OKAWA
  */
sealed trait Distance extends ((Vector, Vector) => Double) with Serializable {

  /**
    * Measures the distance between two vectors.
    *
    * @param lhs a left-hand-side vector to be measures
    * @param rhs a right-hand-side vector to be measured
    * @return a distance measure between the two vectors
    */
  protected def measure(lhs: Vector, rhs: Vector): Double

  /**
    * Caluculates the distance measure between two vectors with additional validations.
    *
    * @param lhs a left-hand-side vector to be caluculated
    * @param rhs a right-hand-side vector to be caluculated
    * @return a distance measure between the two vectors
    */
  def apply(lhs: Vector, rhs: Vector): Double = {
    val distance = measure(lhs, rhs)
    require(distance >= 0.0)
    distance
  }
}

object Distance {

  /** Instanciates the `Distance` from the specified string key. */
  def apply(value: String): Distance = value.toLowerCase match {
    case "cosine" => CosineDistance
    case "euclidean" => EuclideanDistance
    case "manhattan" => ManhattanDistance
    case "hamming" => HammingDistance
    case "jaccard" => JaccardDistance
    case _ =>
      throw new IllegalArgumentException(
        s"key is not defined as distance: $value")
  }
}

/**
  * CosineDistance represents the cosine distance measurement between two vectors.
  *
  * @author Shingo OKAWA
  */
final case object CosineDistance extends Distance {

  /**
    * Measures the distance between two vectors.
    *
    * @param lhs a left-hand-side vector to be measures
    * @param rhs a right-hand-side vector to be measured
    * @return a distance measure between the two vectors
    */
  override def measure(lhs: Vector, rhs: Vector): Double = {
    val result = 1.0 - (math.abs(BLAS.dot(lhs, rhs)) / (Vectors.norm(lhs, 2) * Vectors
      .norm(rhs, 2)))
    if (result < 0.0) 0.0
    else result
  }
}

/**
  * EuclideanDistance represents the euclidean distance measurement between two vectors.
  *
  * @author Shingo OKAWA
  */
final case object EuclideanDistance extends Distance {

  /**
    * Measures the distance between two vectors.
    *
    * @param lhs a left-hand-side vector to be measures
    * @param rhs a right-hand-side vector to be measured
    * @return a distance measure between the two vectors
    */
  override def measure(lhs: Vector, rhs: Vector): Double =
    norm(lhs.asBreeze - rhs.asBreeze, 2.0)
}

/**
  * ManhattanDistance represents the manhattan distance measurement between two vectors.
  *
  * @author Shingo OKAWA
  */
final case object ManhattanDistance extends Distance {

  /**
    * Measures the distance between two vectors.
    *
    * @param lhs a left-hand-side vector to be measures
    * @param rhs a right-hand-side vector to be measured
    * @return a distance measure between the two vectors
    */
  override def measure(lhs: Vector, rhs: Vector): Double =
    norm(lhs.asBreeze - rhs.asBreeze, 1.0)
}

/**
  * HammingDistance represents the hamming distance measurement between two vectors.
  *
  * @author Shingo OKAWA
  */
final case object HammingDistance extends Distance {

  /**
    * Measures the distance measure between two vectors.
    *
    * @param lhs a left-hand-side vector to be caluculated
    * @param rhs a right-hand-side vector to be caluculated
    * @return a similarity measure between the two vectors
    */
  override def measure(lhs: Vector, rhs: Vector): Double = {
    val (li, ri) = (lhs.toSparse.indices.toSet, rhs.toSparse.indices.toSet)
    ((li union ri).size - (li intersect ri).size).toDouble
  }
}

/**
  * JaccardDistance represents the jaccard distance measurement between two vectors.
  *
  * @author Shingo OKAWA
  */
final case object JaccardDistance extends Distance {

  /**
    * Measures the distance measure between two vectors.
    *
    * @param lhs a left-hand-side vector to be caluculated
    * @param rhs a right-hand-side vector to be caluculated
    * @return a similarity measure between the two vectors
    */
  override def measure(lhs: Vector, rhs: Vector): Double = {
    val (li, ri) = (lhs.toSparse.indices.toSet, rhs.toSparse.indices.toSet)
    1.0 - ((li intersect ri).size / (li union ri).size.toDouble)
  }
}
