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
  * PropSpec for [[org.apache.spark.ml.util.interval.Bound]].
  *
  * @author Shingo OKAWA
  */
class BoundsTest
    extends IntervalPropSpec
    with GeneratorDrivenPropertyChecks
    with Matchers {
  property("a lower bound contains a lower bound that starts before") {
    forAll { (x: Endpoint, y: Endpoint) =>
      whenever(x isBelow y) {
        (LowerBound(x) contains LowerBound(y)) should be(true)
      }
    }
  }

  property("an upper bound contains an upper bound that starts after") {
    forAll { (x: Endpoint, y: Endpoint) =>
      whenever(x isAbove y) {
        (UpperBound(x) contains UpperBound(y)) should be(true)
      }
    }
  }
}
