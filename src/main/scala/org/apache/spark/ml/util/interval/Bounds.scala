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
  * A `Bound` is composed of a single endpoint and a semi-unbounded, continuous, infinite, total ordered set. The
  * `Bound` may either point in the lower direction, towards smalle values, or in the upper direction, towards
  * larger values.
  *
  * @author Shingo OKAWA
  */
sealed abstract class Bound extends DoublePredicate with Serializable {

  /**
    * Returns the associated endpoint.
    */
  def endpoint: Endpoint

  /**
    * Returns `true` if this bound contains the other.
    */
  def contains(that: Bound): Boolean

  /**
    * Returns `true` if this bound intersects the other.
    */
  def intersects(that: Bound): Boolean
}

private[interval] case class LowerBound(val endpoint: Endpoint)
    extends Bound
    with Ordered[LowerBound] {

  /**
    * Returns `true` if the given `point` is contained in this bound.
    */
  override def apply(point: Double): Boolean = endpoint match {
    case Closed(boundary) => point >= boundary
    case Open(boundary) => point > boundary
    case Unbounded() => true
  }

  /**
    * Returns `true` if this bound contains the other.
    */
  override def contains(that: Bound): Boolean =
    if (this.endpoint.isUnbounded) true
    else
      that match {
        case LowerBound(thatEndpoint) => this.endpoint isBelow thatEndpoint
        case _ => false
      }

  /**
    * Returns `true` if this bound intersects the other.
    */
  override def intersects(that: Bound): Boolean =
    if (this.endpoint.isUnbounded || that.endpoint.isUnbounded) true
    else
      that match {
        case LowerBound(_) => true
        case _ =>
          (this.endpoint, that.endpoint) match {
            case (Closed(l), Closed(r)) => l <= r
            case (Closed(l), Open(r)) => l < r
            case (Open(l), Closed(r)) => l < r
            case (Open(l), Open(r)) => l < r
            case _ => throw new AssertionError()
          }
      }

  /**
    * Result of comparing this with operand that. Returns x where x < 0 iff this < that, x == 0 iff this == that
    * and x > 0 iff this > that.
    */
  override def compare(that: LowerBound): Int =
    if (this == that) 0
    else if (this contains that) -1
    else 1

  /**
    * Returns the string representation of this bound.
    */
  override def toString: String = endpoint match {
    case Closed(position) => s"[$position"
    case Open(position) => s"($position"
    case Unbounded() => "(inf"
  }
}

private[interval] case class UpperBound(val endpoint: Endpoint)
    extends Bound
    with Ordered[UpperBound] {

  /**
    * Returns `true` if the given `point` is contained in this bound.
    */
  override def apply(point: Double): Boolean = endpoint match {
    case Closed(boundary) => point <= boundary
    case Open(boundary) => point < boundary
    case Unbounded() => true
  }

  /**
    * Returns `true` if this bound contains the other.
    */
  override def contains(that: Bound): Boolean =
    if (this.endpoint.isUnbounded) true
    else
      that match {
        case UpperBound(thatEndpoint) => this.endpoint isAbove thatEndpoint
        case _ => false
      }

  /**
    * Returns `true` if this bound intersects the other.
    */
  override def intersects(that: Bound): Boolean =
    if (this.endpoint.isUnbounded || that.endpoint.isUnbounded) true
    else
      that match {
        case UpperBound(_) => true
        case _ =>
          (this.endpoint, that.endpoint) match {
            case (Closed(l), Closed(r)) => l >= r
            case (Closed(l), Open(r)) => l > r
            case (Open(l), Closed(r)) => l > r
            case (Open(l), Open(r)) => l > r
            case _ => throw new AssertionError()
          }
      }

  /**
    * Result of comparing this with operand that. Returns x where x < 0 iff this < that, x == 0 iff this == that
    * and x > 0 iff this > that.
    */
  override def compare(that: UpperBound): Int =
    if (this == that) 0
    else if (this contains that) 1
    else -1

  /**
    * Returns the string representation of this bound.
    */
  override def toString: String = endpoint match {
    case Closed(position) => s"$position]"
    case Open(position) => s"$position)"
    case Unbounded() => "inf)"
  }
}
