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

import scala.reflect.ClassTag
import org.scalacheck.{Arbitrary, Gen}
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.PropSpec

/**
  * PropSpec base for interval package.
  *
  * @author Shingo OKAWA
  */
abstract class IntervalPropSpec extends PropSpec {
  implicit def arbitraryOpenEndpoint: Arbitrary[Open] =
    Arbitrary { for (at <- arbitrary[Double]) yield Open(at) }

  implicit def arbitraryClosedEndpoint: Arbitrary[Closed] =
    Arbitrary { for (at <- arbitrary[Double]) yield Closed(at) }

  implicit def arbitraryUnboundedEndpoint: Arbitrary[Unbounded] =
    Arbitrary(Unbounded())

  implicit def arbitraryEndpoint: Arbitrary[Endpoint] =
    Arbitrary(
      Gen.frequency(
        4 -> arbitrary[Open],
        4 -> arbitrary[Closed],
        2 -> arbitrary[Unbounded]
      ))

  implicit def arbitraryLowerBound: Arbitrary[LowerBound] =
    Arbitrary(for (endpoint <- arbitrary[Endpoint]) yield LowerBound(endpoint))

  implicit def arbitraryUpperBound: Arbitrary[UpperBound] =
    Arbitrary(for (endpoint <- arbitrary[Endpoint]) yield UpperBound(endpoint))

  implicit def arbitraryBound: Arbitrary[Bound] =
    Arbitrary(Gen.oneOf(arbitrary[LowerBound], arbitrary[UpperBound]))

  implicit def arbitraryInterval: Arbitrary[Interval] = Arbitrary {
    def validate(lhs: Endpoint, rhs: Endpoint) =
      Interval.validate(LowerBound(lhs), UpperBound(rhs)) || Interval
        .validate(LowerBound(rhs), UpperBound(lhs))
    def interval(lhs: Endpoint, rhs: Endpoint) =
      if (Interval.validate(LowerBound(lhs), UpperBound(rhs)))
        new Interval(LowerBound(lhs), UpperBound(rhs))
      else new Interval(LowerBound(rhs), UpperBound(lhs))
    for {
      x <- arbitrary[Endpoint]
      y <- arbitrary[Endpoint] if validate(x, y)
    } yield interval(x, y)
  }

  implicit def arbitrary2DimensionalCube: Arbitrary[Cube] = Arbitrary {
    for {
      x <- arbitrary[Interval]
      y <- arbitrary[Interval]
    } yield Cube(x, y)
  }
}
