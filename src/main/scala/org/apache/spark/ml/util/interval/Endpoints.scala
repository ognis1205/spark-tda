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
  * An `Endpoint` is a point that bounds an interval. An `Endpoint` can be closed, open or unbounded.
  *
  * @author Shingo OKAWA
  */
sealed abstract class Endpoint extends Serializable {

  /**
    * Returns `true` if all points below the other endpoint are also below this endpoint.
    */
  def isAbove(that: Endpoint): Boolean

  /**
    * Returns `true` if all points above the other endpoint are also above this endpoint.
    */
  def isBelow(that: Endpoint): Boolean

  /**
    * Returns `true` if this endpoint is bounded.
    */
  def isBounded: Boolean

  /**
    * Returns `true` if this endpoint is unbounded.
    */
  def isUnbounded: Boolean = !this.isBounded

  /**
    * Transforms this bound.
    */
  def map(func: Double => Double): Endpoint
}

private[interval] case class Closed(val position: Double) extends Endpoint {

  /**
    * Returns `true` if all points below the other endpoint are also below this endpoint.
    */
  override def isAbove(that: Endpoint): Boolean = that match {
    case Open(thatPosition) => this.position >= thatPosition
    case Closed(thatPosition) => this.position >= thatPosition
    case Unbounded() => false
  }

  /**
    * Returns `true` if all points above the other bound are also above this bound.
    */
  override def isBelow(that: Endpoint): Boolean = that match {
    case Open(thatPosition) => this.position <= thatPosition
    case Closed(thatPosition) => this.position <= thatPosition
    case Unbounded() => false
  }

  /**
    * Returns `true` if this endpoint is bounded.
    */
  override def isBounded: Boolean = true

  /**
    * Transforms this bound.
    */
  override def map(func: Double => Double): Endpoint = Closed(func(position))
}

private[interval] case class Open(val position: Double) extends Endpoint {

  /**
    * Returns `true` if all points below the other endpoint are also below this endpoint.
    */
  override def isAbove(that: Endpoint): Boolean = that match {
    case Open(thatPosition) => this.position >= thatPosition
    case Closed(thatPosition) => this.position > thatPosition
    case Unbounded() => false
  }

  /**
    * Returns `true` if all points above the other bound are also above this bound.
    */
  override def isBelow(that: Endpoint): Boolean = that match {
    case Open(thatPosition) => this.position <= thatPosition
    case Closed(thatPosition) => this.position < thatPosition
    case Unbounded() => false
  }

  /**
    * Returns `true` if this endpoint is bounded.
    */
  override def isBounded: Boolean = true

  /**
    * Transforms this bound.
    */
  override def map(func: Double => Double): Endpoint = Open(func(position))
}

private[interval] case class Unbounded() extends Endpoint {

  /**
    * Returns `true` if all points below the other endpoint are also below this endpoint.
    */
  override def isAbove(that: Endpoint): Boolean = true

  /**
    * Returns `true` if all points above the other bound are also above this bound.
    */
  override def isBelow(that: Endpoint): Boolean = true

  /**
    * Returns `true` if this endpoint is bounded.
    */
  override def isBounded: Boolean = false

  /**
    * Transforms this bound.
    */
  override def map(func: Double => Double): Endpoint = Unbounded()
}
