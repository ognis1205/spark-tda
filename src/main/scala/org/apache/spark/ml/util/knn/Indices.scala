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

import scala.annotation.tailrec

/**
  * An index from keys of type `K` to values of type `Int`.
  *
  * @author Shingo OKAWA
  */
private[knn] trait Index[K] extends IndexLike[K, Index[K]] with Serializable {

  /** Instanciates empty index. */
  def empty: Index[K] = Index.empty[K]
}

private[knn] object Index {

  /** Instanciates empty index. */
  def empty[K]: Index[K] = EmptyIndex.asInstanceOf[Index[K]]

  private object EmptyIndex extends Index[Any] with Serializable {

    /** Optionally returns the value associated with a key. */
    override def get(key: Any): Option[Nothing] = None

    /** Returns the value associated with a key, or a default value if the key is not contained in the index. */
    override def getOrElse(key: Any, default: => Int): Int = default

    /**
      * Retrieves the value which is associated with the given key. This method invokes the `default` method of the
      * map if there is no mapping from the given key to a value. Unless overridden, the `default` method throws a
      * `NoSuchElementException`.
      */
    override def apply(key: Any): Int =
      throw new NoSuchElementException(s"key not found: $key")

    /** Returns the total number of the underlying indices. */
    override def numIndices: Int = 0

    /** Tests whether this index is defined at a specified key. */
    override def isDefinedAt(key: Any) = false
  }
}

/**
  * A template trait for immutable indices.
  *
  * @author Shingo OKAWA
  */
private[knn] trait IndexLike[K, +This <: IndexLike[K, This] with Index[K]] {
  self =>

  /** Optionally returns the value associated with a key. */
  def get(key: K): Option[Int]

  /** Returns the value associated with a key, or a default value if the key is not contained in the index. */
  def getOrElse(key: K, default: => Int): Int = get(key) match {
    case Some(v) => v
    case None => default
  }

  /**
    * Retrieves the value which is associated with the given key. This method invokes the `default` method of the
    * map if there is no mapping from the given key to a value. Unless overridden, the `default` method throws a
    * `NoSuchElementException`.
    */
  def apply(key: K): Int = get(key) match {
    case None => default(key)
    case Some(value) => value
  }

  /** Returns the total number of the underlying indices. */
  def numIndices: Int

  /**
    * Defines the default value computation for the index, returned when a key is not found. The method implemented
    * here throws an exception, but it might be overridden in subclasses.
    */
  def default(key: K): Int =
    throw new NoSuchElementException(s"key not found: $key")

  /** Tests whether this index is defined at a specified key. */
  def isDefinedAt(key: K): Boolean = get(key).isDefined
}

/**
  * A `TopTrees` is used to index data points for k-NN search. It represents a forest of binary tree nodes.
  * This implementation is based on the following report:
  * [[https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/32616.pdf
  *   Clustering billions of images with large scale nearest neighbor search]]
  *
  * @see https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/32616.pdf
  * @author Shingo OKAWA
  */
private[ml] final class TopTrees private (
    val trees: Map[Int, TopTrees.TreeEntry])
    extends Index[(Int, VectorWithId)]
    with IndexLike[(Int, VectorWithId), TopTrees]
    with Serializable {

  /** Optionally returns the value associated with a key. */
  override def get(key: (Int, VectorWithId)): Option[Int] =
    trees.get(key._1).map { entry =>
      search(key._2, entry.tree, entry.indexStartsFrom)
    }

  /** Returns the total number of the underlying indices. */
  override def numIndices: Int = trees.values.foldLeft(0) {
    case (acc, entry) => acc + entry.tree.numLeaves
  }

  /**
    * Searches leaf index to partition data points.
    *
    * @param query vector with its id to partition
    * @param tree top tree constructed using sampled data points
    * @param acc accumulator used to determine leaf index
    * @return leaf/partition index
    */
  @tailrec private def search(query: VectorWithId, tree: Tree, acc: Int): Int =
    tree match {
      case node: Branch =>
        val d = node.measure(node.vantagePoint.vector, query.vector)
        if (d < node.vantagePoint.median) search(query, node.left, acc)
        else search(query, node.right, acc + node.left.numLeaves)
      case _ => acc
    }
}

private[ml] object TopTrees {

  /** An `TreeEntry` represents indices for each cover segements. */
  private[knn] final case class TreeEntry(val indexStartsFrom: Int,
                                          val tree: Tree)

  /** Returns empty `TopTrees`. */
  def empty = new TopTrees(Map.empty[Int, TreeEntry])

  /** Instanciates `TopTrees` from the given `Tree`s. */
  def fromSeq(buf: Seq[(Int, Tree)]): TopTrees = {
    val indices = buf.scanLeft(0) {
      case (acc, tree) => acc + tree._2.numLeaves
    }
    val entries = buf zip indices map {
      case (tree, index) =>
        (tree._1, TreeEntry(index, tree._2))
    }
    new TopTrees(Map(entries map { e =>
      e._1 -> e._2
    }: _*))
  }

  /** Instanciates `TopTrees`. */
  def apply(buf: Seq[(Int, Tree)]): TopTrees = fromSeq(buf)
}
