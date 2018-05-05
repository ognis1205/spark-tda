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
package org.apache.spark.ml.util.knn

import org.apache.spark.Partitioner

/**
  * A `Partitioner` used to map data vectors to leaf nodes which determines the partition it goes to.
  *
  * @author Shingo OKAWA
  */
private[ml] final class TopTreesPartitioner(topTrees: TopTrees)
    extends Partitioner {

  /** Returns the total number of partitions. */
  override def numPartitions: Int = topTrees.numIndices

  /** Returns the partition identifier for the given key. */
  override def getPartition(key: Any): Int = key match {
    case (coverId: Int, vector: VectorWithId) => topTrees((coverId, vector))
    case _ =>
      throw new IllegalArgumentException(
        s"key must be of type (Int, VectorWithId) but got : $key")
  }
}
