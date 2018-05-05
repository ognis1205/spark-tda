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
import org.scalatest.GivenWhenThen
import org.scalatest.prop.GeneratorDrivenPropertyChecks

/**
  * PropSpec for [[org.apache.spark.ml.util.interval.Cube]].
  *
  * @author Shingo OKAWA
  */
class CubeTest
    extends IntervalPropSpec
    with GeneratorDrivenPropertyChecks
    with GivenWhenThen
    with Matchers {
  property("cube intersection is commutative") {
    forAll { (lhs: Cube, rhs: Cube) =>
      (lhs intersects rhs) should equal(rhs intersects lhs)
    }
  }

  property("cube inclusion is transitive") {
    forAll { (lhs: Cube, rhs: Cube, x: Double, y: Double) =>
      !(lhs contains rhs) || !(rhs contains (x, y)) || (lhs contains (x, y))
    }
  }
}
