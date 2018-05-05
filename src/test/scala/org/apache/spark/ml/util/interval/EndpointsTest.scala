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
  * PropSpec for [[org.apache.spark.ml.util.interval.Endpoint]].
  *
  * @author Shingo OKAWA
  */
class EndpointsTest
    extends IntervalPropSpec
    with GeneratorDrivenPropertyChecks
    with Matchers {
  property("the above relation on endpoints is transitive") {
    forAll { (x: Endpoint, y: Endpoint, z: Endpoint) =>
      val (u, l) = if (x isAbove y) (x, y) else (y, x)
      whenever(l isAbove z) { (u isAbove z) should be(true) }
    }
  }

  property("the below relation on endpoints is transitive") {
    forAll { (x: Endpoint, y: Endpoint, z: Endpoint) =>
      val (l, u) = if (x isBelow y) (x, y) else (y, x)
      whenever(u isBelow z) { (l isBelow z) should be(true) }
    }
  }
}
