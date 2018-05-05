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
package org.apache.spark.mllib.linalg.distributed.impl

import org.scalacheck.Gen.{choose, oneOf, listOfN}
import org.scalacheck.Arbitrary.arbitrary
import org.scalatest.Matchers
import org.scalatest.prop.GeneratorDrivenPropertyChecks
import org.apache.spark.mllib.linalg.DenseVector

/**
  * FunSpec for [[org.apache.spark.mllib.linalg.distributed.impl.HashFunctions]].
  *
  * @author Shingo OKAWA
  */
class HashFunctionsTest
    extends ImplPropSpec
    with GeneratorDrivenPropertyChecks
    with Matchers {
  import org.scalactic.Tolerance._

  property(
    "simhash returns hashed vector whose dimension is at most the specified signature length") {
    forAll(simhashGen) {
      case (vector, signatureLength, simhash) =>
        val bucket = simhash(0L, 0, vector)
        assert(bucket === simhash(0L, 0, vector.toSparse))
        assert(bucket.signature.length <= signatureLength)
    }
  }

  property(
    "minhash returns hashed vector whose dimension is the specified signature length") {
    forAll(minhashGen) {
      case (vector, signatureLength, minhash) =>
        val bucket = minhash(0L, 0, vector)
        assert(bucket === minhash(0L, 0, vector.toSparse))
        assert(bucket.signature.length === signatureLength)
    }
  }

  property(
    "pstable returns hashed vector whose dimension is the specified signature length") {
    forAll(pstableGen) {
      case (vector, signatureLength, pstableL1, pstableL2) =>
        val bucketL1 = pstableL1(0L, 0, vector)
        val bucketL2 = pstableL2(0L, 0, vector)
        assert(bucketL1 === pstableL1(0L, 0, vector.toSparse))
        assert(bucketL2 === pstableL2(0L, 0, vector.toSparse))
        assert(bucketL1.signature.length === signatureLength)
        assert(bucketL2.signature.length === signatureLength)
    }
  }

  property(
    "bit sampling returns hashed vector whose dimension is at most the specified signature length") {
    forAll(bsampleGen) {
      case (vector, signatureLength, bsample) =>
        val bucket = bsample(0L, 0, vector)
        assert(bucket === bsample(0L, 0, vector.toSparse))
        assert(bucket.signature.length <= signatureLength)
    }
  }
}
