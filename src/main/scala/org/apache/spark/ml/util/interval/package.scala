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
package org.apache.spark.ml.util

/**
  * Collection of interval utilities used by SparkTDA MLLib.
  */
package object interval {

  /**
    * An `DoublePredicate` represents predicates on set of `Double`s. All interval related instances can be
    * represented as an `DoublePredicate`.
    */
  private[interval] abstract class DoublePredicate
      extends (Double => Boolean) {

    /** Returns `true` if this predicate contains the point. */
    def contains(x: Double): Boolean = this(x)
  }

  /** Tabulates all combinations over the specified `Seq`s. */
  private def cartesian[T](seq: Seq[Seq[T]]): Seq[Seq[T]] =
    seq match {
      case Seq(x) => x map (Seq(_))
      case Seq(x, xs @ _ *) =>
        for {
          i <- x
          j <- cartesian(xs)
        } yield Seq(i) ++ j
    }

  /** Splits an "interval" into the specified number of subintervals. */
  private def split(
      lower: Double,
      upper: Double,
      numberOfSplits: Int
  ): Seq[Tuple2[Double, Double]] = {
    require(upper > lower && numberOfSplits > 0)
    val width = (upper - lower) / numberOfSplits
    (0 until numberOfSplits).foldLeft(Seq.empty[Tuple2[Double, Double]]) {
      case (acc, i) =>
        val left = lower + i * width
        acc :+ (left, left + width)
    }
  }

  /**
    * Instanciates an closed cover of the specified cartesian product.
    *
    * @param numberOfSplits the number of subintervals within each dimension
    * @param overlapRation the ratio of length where overlaps to neighboring intervals
    * @param ranges lower-upper bound pairs of each dimension
    * @return closed cubes which covers the specified hypercube
    */
  private[ml] def ClosedCover(
      numberOfSplits: Int,
      overlapRatio: Double,
      ranges: Seq[Tuple2[Double, Double]]): Seq[Cube] = {
    require(overlapRatio >= 0.0)
    cartesian(ranges.map(r => split(r._1, r._2, numberOfSplits))).zipWithIndex
      .foldLeft(Seq.empty[Cube]) {
        case (acc, (bounds, idx)) =>
          acc :+ Cube(idx, bounds.map(b => Interval.closed(b._1, b._2)): _*)
      }
      .map { c =>
        c.map { i: Interval =>
          i.length match {
            case Some(l) =>
              i.map(lower => lower - (l * overlapRatio / 2.0),
                    upper => upper + (l * overlapRatio / 2.0))
            case None => i
          }
        }
      }
  }

  /**
    * Instanciates an open cover of the specified cartesian product.
    *
    * @param numberOfSplits the number of subintervals within each dimension
    * @param overlapRation the ratio of length where overlaps to neighboring intervals
    * @param ranges lower-upper bound pairs of each dimension
    * @return open cubes which covers the specified hypercube
    */
  private[ml] def OpenCover(numberOfSplits: Int,
                            overlapRatio: Double,
                            ranges: Seq[Tuple2[Double, Double]]): Seq[Cube] = {
    require(overlapRatio >= 0.0)
    cartesian(ranges.map(r => split(r._1, r._2, numberOfSplits))).zipWithIndex
      .foldLeft(Seq.empty[Cube]) {
        case (acc, (bounds, idx)) =>
          acc :+ Cube(idx, bounds.map(b => Interval.open(b._1, b._2)): _*)
      }
      .map { c =>
        c.map { i: Interval =>
          i.length match {
            case Some(l) =>
              i.map(lower => lower - (l * overlapRatio / 2.0),
                    upper => upper + (l * overlapRatio / 2.0))
            case None => i
          }
        }
      }
  }
}
