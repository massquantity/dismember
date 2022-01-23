package com.mass.sparkdl.nn.graphnn

import scala.collection.mutable.ArrayBuffer

private[sparkdl] class Node[T](val element: T) extends Serializable { self =>

  private val nexts = new ArrayBuffer[(Node[T], Edge)]()
  private val prevs = new ArrayBuffer[(Node[T], Edge)]()

  override def toString: String = s"(${element.toString})"

  private var nodeName: String = _

  def getName: String = nodeName

  def setName(name: String): Unit = {
    nodeName = name
  }

  def find(name: String): Option[Node[T]] = {
    nexts.find(_._1.nodeName == name) match {
      case Some(i) => Some(i._1)
      case _ => None
    }
  }

  def nextNodes: Seq[Node[T]] = nexts.map(_._1).toSeq

  def nextEdges: Seq[Edge] = nexts.map(_._2).toSeq

  def nextNodesAndEdges: Seq[(Node[T], Edge)] = nexts.toSeq

  def prevNodes: Seq[Node[T]] = prevs.map(_._1).toSeq

  def prevEdges: Seq[Edge] = prevs.map(_._2).toSeq

  def prevNodesAndEdges: Seq[(Node[T], Edge)] = prevs.toSeq

  def ->(node: Node[T]): Node[T] = {
    this.add(node)
  }

  def add(node: Node[T], e: Edge = Edge()): Node[T] = {
    if (!node.prevs.contains((this, e))) {
      node.prevs += Tuple2(this, e)
    }
    if (!this.nexts.contains((node, e))) {
      this.nexts += Tuple2(node, e)
    }
    node
  }

  def addPrevious(node: Node[T], e: Edge = Edge()): Unit = {
    if (!this.prevs.contains((node, e))) {
      this.prevs += Tuple2(node, e)
    }
  }

  def addNexts(node: Node[T], e: Edge = Edge()): Unit = {
    if (!this.nexts.contains((node, e))) {
      this.nexts += Tuple2(node, e)
    }
  }

  def from(node: Node[T], e: Edge = Edge()): Node[T] = {
    if (!node.nexts.contains((this, e))) {
      node.nexts += Tuple2(this, e)
    }
    if (!this.prevs.contains((node, e))) {
      this.prevs += Tuple2(node, e)
    }
    node
  }

  def delete(node: Node[T], e: Edge = null): Node[T] = {
    if (e != null) {
      if (node.prevs.contains((this, e))) {
        node.prevs -= Tuple2(this, e)
      }
      if (this.nexts.contains((node, e))) {
        this.nexts -= Tuple2(node, e)
      }
    } else {
      node.prevs.filter(_._1 == self).foreach(node.prevs -= _)
      this.nexts.filter(_._1 == node).foreach(this.nexts -= _)
    }
    this
  }

  def apply[M](meta: M): (this.type, M) = {
    (this, meta)
  }

  def removePrevEdges(): Node[T] = {
    for {
      prev <- prevs
      p_nexts = prev._1.nexts
      pn <- p_nexts
      if pn._1 == self
    } p_nexts -= pn

    prevs.clear()
    this
  }

  def removeNextEdges(): Node[T] = {
    for {
      next <- nexts
      n_prevs = next._1.prevs
      pn <- n_prevs
      if pn._1 == self
    } n_prevs -= pn

    nexts.clear()
    this
  }

  def graph(reverse: Boolean = false): DirectedGraph[T] = {
    new DirectedGraph[T](this, reverse)
  }
}

object Node {
  def apply[T](element: T): Node[T] = new Node(element)
}
