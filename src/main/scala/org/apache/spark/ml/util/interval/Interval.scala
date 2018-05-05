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

/**
  * An `Interval` contains all values between its lower and upper bound. The lower and/or upper bound could be
  * unbounded.
  *
  * @param lower lower bound of the interval
  * @param upper upper bound of the interval
  *
  * @author Shingo OKAWA
  */
private[interval] final case class Interval(val lower: LowerBound,
                                            val upper: UpperBound)
    extends DoublePredicate
    with Ordered[Interval]
    with Serializable {

  require(Interval.validate(lower, upper))

  /**
    * Returns `true` if this interval contains the point.
    */
  override def apply(x: Double): Boolean =
    (this.lower contains x) && (this.upper contains x)

  /**
    * Result of comparing this with operand that. Returns x where x < 0 iff this < that, x == 0 iff this == that
    * and x > 0 iff this > that.
    */
  override def compare(that: Interval): Int =
    this.lower compare that.lower match {
      case r if r != 0 => r
      case _ => this.upper compare that.upper
    }

  /**
    * Returns the string representation of this interval.
    */
  override def toString: String =
    this.lower.toString + ", " + this.upper.toString

  /**
    * Returns the length of this interval if it can be defined.
    */
  def length: Option[Double] = {
    (this.lower.endpoint, this.upper.endpoint) match {
      case (Closed(l), Closed(r)) => Some(r - l)
      case (Closed(l), Open(r)) => Some(r - l)
      case (Open(l), Closed(r)) => Some(r - l)
      case (Open(l), Open(r)) => Some(r - l)
      case _ => None
    }
  }

  /**
    * Returns `true` if this interval contains the other.
    */
  def contains(that: Interval): Boolean =
    (this.lower contains that.lower) && (this.upper contains that.upper)

  /**
    * Returns `true` if this interval intersects the other.
    */
  def intersects(that: Interval): Boolean =
    (this.lower intersects that.upper) && (this.upper intersects that.lower)

  /**
    * Transform this interval.
    */
  def map(func: Double => Double): Interval =
    Interval(LowerBound(this.lower.endpoint.map(func)),
             UpperBound(this.upper.endpoint.map(func)))

  /**
    * Transform this interval.
    */
  def map(lf: Double => Double, uf: Double => Double): Interval =
    Interval(LowerBound(this.lower.endpoint.map(lf)),
             UpperBound(this.upper.endpoint.map(uf)))
}

private[interval] object Interval {

  /**
    * Returns `true` if the specified lower bound and upper bound intersect properly.
    */
  private[interval] def validate(lower: LowerBound,
                                 upper: UpperBound): Boolean =
    lower intersects upper

  /**
    * Instanciates an open interval which has specified lower and upper bound.
    */
  def open(lower: Double, upper: Double): Interval =
    Interval(LowerBound(Open(lower)), UpperBound(Open(upper)))

  /**
    * Instanciates a closed interval which has specified lower and upper bound.
    */
  def closed(lower: Double, upper: Double): Interval =
    Interval(LowerBound(Closed(lower)), UpperBound(Closed(upper)))

  /**
    * Instanciates a left-open right-closed interval which has specified lower and upper bound.
    */
  def openClosed(lower: Double, upper: Double): Interval =
    Interval(LowerBound(Open(lower)), UpperBound(Closed(upper)))

  /**
    * Instanciates a right-open left-closed interval which has specified lower and upper bound.
    */
  def closedOpen(lower: Double, upper: Double): Interval =
    Interval(LowerBound(Closed(lower)), UpperBound(Open(upper)))

  /**
    * Instanciates left-open-bounded interval which has specified lower bound.
    */
  def greaterThan(lower: Double): Interval =
    Interval(LowerBound(Open(lower)), UpperBound(Unbounded()))

  /**
    * Instanciates left-closed-bounded interval which has specified lower bound.
    */
  def greaterEqual(lower: Double): Interval =
    Interval(LowerBound(Closed(lower)), UpperBound(Unbounded()))

  /**
    * Instanciates right-open-bounded interval which has specified lower bound.
    */
  def lessThan(upper: Double): Interval =
    Interval(LowerBound(Unbounded()), UpperBound(Open(upper)))

  /**
    * Instanciates right-closed-bounded interval which has specified lower bound.
    */
  def lessEqual(upper: Double): Interval =
    Interval(LowerBound(Unbounded()), UpperBound(Closed(upper)))
}
