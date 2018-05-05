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

import scala.language.implicitConversions
import org.apache.spark.annotation.DeveloperApi

/**
  * :: DeveloperApi ::
  * SparkTDA specific [[CoordinateMatrix]] functions.
  */
@DeveloperApi
class CoordinateMatrixFunctions(self: CoordinateMatrix) extends Serializable {

  /**
    * Caluculates Hadamard product of specified [[CoordinateMatrix]]s.
    *
    * @param other the right-hand-side [[CoordinateMatrix]]
    * @param func the resulting element-wise product values will be transformed by this function
    * @return hadamard product
    */
  def hproduct(other: CoordinateMatrix,
               func: Double => Double = (x => x)): CoordinateMatrix =
    new CoordinateMatrix(
      self.entries
        .map(e => ((e.i, e.j), e.value))
        .join(other.entries.map(e => ((e.i, e.j), e.value)))
        .map {
          case ((i, j), (lhv, rhv)) =>
            new MatrixEntry(i, j, func(lhv * rhv))
        },
      math.max(self.numRows, other.numRows),
      math.max(self.numCols, other.numCols)
    )
}

@DeveloperApi
object CoordinateMatrixFunctions {

  /** Implicit conversion from a CoordinateMatrix to CoordinateMatrixFunctions. */
  implicit def fromCoordinateMatrix(
      matrix: CoordinateMatrix): CoordinateMatrixFunctions =
    new CoordinateMatrixFunctions(matrix)
}
