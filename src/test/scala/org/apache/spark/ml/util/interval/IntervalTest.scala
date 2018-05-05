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

import org.scalatest.Matchers
import org.scalatest.prop.GeneratorDrivenPropertyChecks

/**
  * PropSpec for [[org.apache.spark.ml.util.interval.Interval]].
  *
  * @author Shingo OKAWA
  */
class IntervalTest
    extends IntervalPropSpec
    with GeneratorDrivenPropertyChecks
    with Matchers {
  property("intervals intersect if they overlap") {
    forAll { (a: Double, b: Double, c: Double, d: Double) =>
      val es = List(a, b, c, d).sorted
      val la = es(0)
      val lb = es(1)
      val ua = es(2)
      val ub = es(3)
      Interval.greaterEqual(la) intersects Interval.closed(lb, ub) should be(
        true)
      Interval.closed(la, ua) intersects Interval.closed(lb, ub) should be(
        true)
      whenever(lb < ua) {
        whenever(la < ua) {
          Interval.open(la, ua) intersects Interval.closed(lb, ub) should be(
            true)
        }
        whenever(lb < ub) {
          Interval.closed(la, ua) intersects Interval.open(lb, ub) should be(
            true)
        }
        whenever(la < ua && lb < ub) {
          Interval.open(la, ua) intersects Interval.open(lb, ub) should be(
            true)
        }
      }
    }
  }

  property("interval intersection is commutative") {
    forAll { (lhs: Interval, rhs: Interval) =>
      (lhs intersects rhs) should equal(rhs intersects lhs)
    }
  }

  property("interval inclusion is transitive") {
    forAll { (lhs: Interval, rhs: Interval, x: Double) =>
      !(lhs contains rhs) || !(rhs contains x) || (lhs contains x)
    }
  }
}
