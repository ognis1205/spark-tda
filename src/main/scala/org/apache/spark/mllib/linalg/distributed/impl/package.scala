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
package org.apache.spark.mllib.linalg.distributed

import java.util.{Random => JRandom}

/**
  * Collection of utilities used by linalg similarity package.
  */
package object impl {

  /**
    * A random permutation function can be approximated by the following function:
    *   pi(x) := (ax + b) mod p, where a, b are randomly chosen integers and p is a prime number.
    *
    * @see [[http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.121.8215&rep=rep1&type=pdf
    *        Min-wise independent permutations]]
    * @author Shingo OKAWA
    */
  private[impl] class RandomPermutation(
      a: Int,
      b: Int,
      p: Int,
      dimension: Int
  ) extends Serializable {
    implicit class ExtensionOfLong(x: Long) {
      def mod(y: Long): Long = x % y + (if (x < 0) y else 0)
    }

    /**
      * Caluculates the permuted value for the specified integer.
      */
    def apply(x: Int): Int =
      ((((a.longValue * x) + b) mod p) mod dimension).toInt
  }

  private[impl] object RandomPermutation {
    implicit class ExtensionOfJRandom(random: JRandom) {
      def nextPositiveInt(): Int = {
        val next = random.nextInt()
        if (next == 0) nextPositiveInt()
        else next
      }
    }

    /**
      * Generates a random permutation.
      */
    def apply(dimension: Int, prime: Int, random: JRandom): RandomPermutation = {
      val a = random.nextPositiveInt()
      val b = random.nextInt()
      new RandomPermutation(a, b, prime, dimension)
    }
  }
}
