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
package org.apache.spark.ml.util.interval

import scala.collection.IndexedSeqLike
import scala.collection.generic.CanBuildFrom
import scala.collection.mutable.{ArrayBuffer, Builder}

/**
  * A `Cube` is a cartesian product of `Intervals`.
  *
  * @param intervals the underlying intervals of each dimensions
  *
  * @author Shingo OKAWA
  */
private[ml] final class Cube private (val intervals: Array[Interval])
    extends IndexedSeq[Interval]
    with IndexedSeqLike[Interval, Cube]
    with Serializable {

  /**
    * The identifier for this cube.
    */
  var id: Int = null.asInstanceOf[Int]

  /**
    * The default builder for `Cube` objects.
    */
  override protected[this] def newBuilder: Builder[Interval, Cube] =
    Cube.newBuilder

  /**
    * Selects an element by its index in sequence.
    */
  override def apply(idx: Int): Interval = intervals(idx)

  /**
    * The length of sequence
    */
  override def length: Int = intervals.length

  /**
    * Returns the string representation of this cube.
    */
  override def toString: String =
    s"""Cube(id=$id, ${intervals.map(_.toString).mkString(", ")})"""

  /**
    * The dimension of cube.
    */
  def dimension: Int = length

  /**
    * Returns `true` if this cube contains the specified point.
    */
  def contains(x: Seq[Double]): Boolean = {
    require(x.length == this.dimension)
    (x zip this.intervals) forall {
      case (p, i) =>
        i contains p
    }
  }

  /**
    * Returns `true` if this cube contains the other.
    */
  def contains(that: Cube): Boolean = {
    require(this.dimension == that.dimension)
    (this zip that) forall {
      case (lhs, rhs) =>
        lhs contains rhs
    }
  }

  /**
    * Returns `true` if this cube intersects the other.
    */
  def intersects(that: Cube): Boolean = {
    require(this.dimension == that.dimension)
    (this zip that) forall {
      case (lhs, rhs) =>
        lhs intersects rhs
    }
  }

  /**
    * Transform this cube.
    */
  def map(func: Interval => Interval): Cube =
    Cube.fromSeq(id, intervals.map(func))
}

private[ml] object Cube {

  /**
    * Instanciates `Cube` from the given `Interval`s.
    */
  def fromSeq(buf: Seq[Interval]): Cube = {
    val intervals = new Array[Interval](buf.length)
    for (i <- 0 until buf.length) {
      intervals(i) = buf(i)
    }
    new Cube(intervals)
  }

  /**
    * Instanciates `Cube` from the given `Interval`s with specified identifier.
    */
  def fromSeq(id: Int, buf: Seq[Interval]): Cube = {
    val cube = fromSeq(buf)
    cube.id = id
    cube
  }

  /**
    * Instances `Cube` from the given `Interval`s.
    */
  def apply(intervals: Interval*): Cube = fromSeq(intervals.toSeq)

  /**
    * Instanciates `Cube` from the given `Interval`s with specified identifier.
    */
  def apply(id: Int, intervals: Interval*): Cube =
    fromSeq(id, intervals.toSeq)

  /**
    * The default builder for `Cube` objects.
    */
  def newBuilder: Builder[Interval, Cube] =
    new ArrayBuffer mapResult fromSeq

  /**
    * The standard `CanBuildFrom` instance for `Cube` objects.
    */
  @inline
  implicit def canBuildFrom: CanBuildFrom[Cube, Interval, Cube] =
    new CanBuildFrom[Cube, Interval, Cube] {
      override def apply = newBuilder
      override def apply(from: Cube): Builder[Interval, Cube] =
        newBuilder
    }
}
