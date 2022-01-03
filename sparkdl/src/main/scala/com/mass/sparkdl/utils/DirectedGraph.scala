package com.mass.sparkdl.utils

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class DirectedGraph[T](val source: Node[T], val reverse: Boolean = false) extends Serializable {

  def size: Int = this.BFS().size

  def edges: Int = this.BFS().map(_.nextNodes.length).sum

  def topologicalSort(): Array[Node[T]] = {
    val inDegrees = new mutable.LinkedHashMap[Node[T], Int]()
    inDegrees(source) = 0
    DFS().foreach { n =>
      val nextNodes = if (!reverse) n.nextNodes else n.prevNodes
      nextNodes.foreach(m => inDegrees(m) = inDegrees.getOrElse(m, 0) + 1)
    }

    val result = new ArrayBuffer[Node[T]]()
    while (inDegrees.nonEmpty) {
      val startNodes = inDegrees.filterKeys(inDegrees(_) == 0).keySet.toArray
      require(startNodes.nonEmpty, "There exists a cycle in the graph")
      result ++= startNodes
      startNodes.foreach { sn =>
        val nextNodes = if (!reverse) sn.nextNodes else sn.prevNodes
        nextNodes.foreach(n => inDegrees(n) -= 1)
        inDegrees -= sn
      }
    }
    result.toArray
  }

  def DFS(): Iterator[Node[T]] = {
    new Iterator[Node[T]] {
      private val stack = new mutable.Stack[Node[T]]().push(source)
      private val visited = new mutable.HashSet[Node[T]]()

      override def hasNext: Boolean = stack.nonEmpty

      override def next(): Node[T] = {
        require(hasNext)
        val node = stack.pop()
        visited += node
        val nextNodes = if (!reverse) node.nextNodes else node.prevNodes
        // to preserve order
        val nodesSet = new mutable.LinkedHashSet[Node[T]]()
        nextNodes.foreach(nodesSet.add)
        for (n <- nodesSet; if !visited.contains(n) && !stack.contains(n)) {
          stack.push(n)
        }
        node
      }
    }
  }

  def BFS(): Iterator[Node[T]] = {
    new Iterator[Node[T]] {
      private val visited = new mutable.HashSet[Node[T]]()
      private val queue = new mutable.Queue[Node[T]]()
      queue.enqueue(source)

      override def hasNext: Boolean = queue.nonEmpty

      override def next(): Node[T] = {
        require(hasNext)
        val node = queue.dequeue()
        visited.add(node)
        val nextNodes = if (!reverse) node.nextNodes else node.prevNodes
        val nodesSet = new mutable.LinkedHashSet[Node[T]]()
        nextNodes.foreach(nodesSet.add)
        for (n <- nodesSet; if !visited.contains(n) && !queue.contains(n)) {
          queue.enqueue(n)
        }
        node
      }
    }
  }

  def cloneGraph(reverseEdge: Boolean = false): DirectedGraph[T] = {
    val oldToNew = new java.util.HashMap[Node[T], Node[T]]()
    val bfs = BFS().toArray
    bfs.foreach(node => oldToNew.put(node, new Node[T](node.element)))
    bfs.foreach { curNode =>
      if (reverseEdge) {
        curNode.nextNodesAndEdges.foreach { case (node, edge) =>
          if (oldToNew.containsKey(node)) {
            oldToNew.get(curNode).addPrevious(oldToNew.get(node), edge)
          }
        }
        curNode.prevNodesAndEdges.foreach { case (node, edge) =>
          if (oldToNew.containsKey(node)) {
            oldToNew.get(curNode).addNexts(oldToNew.get(node), edge)
          }
        }
      } else {
        curNode.nextNodesAndEdges.foreach { case (node, edge) =>
          if (oldToNew.containsKey(node)) {
            oldToNew.get(curNode).add(oldToNew.get(node), edge)
          }
        }
      }
    }

    if (reverseEdge) {
      new DirectedGraph[T](oldToNew.get(source), !reverse)
    } else {
      new DirectedGraph[T](oldToNew.get(source), reverse)
    }
  }
}

private[sparkdl] class Node[T](var element: T) extends Serializable { self =>

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
    if (!node.prevs.contains((this, e))) node.prevs.append((this, e))
    if (!this.nexts.contains((node, e))) this.nexts.append((node, e))
    node
  }

  def addPrevious(node: Node[T], e: Edge = Edge()): Unit = {
    if (!this.prevs.contains((node, e))) this.prevs.append((node, e))
  }

  def addNexts(node: Node[T], e: Edge = Edge()): Unit = {
    if (!this.nexts.contains((node, e))) this.nexts.append((node, e))
  }

  def from(node: Node[T], e: Edge = Edge()): Node[T] = {
    if (!node.nexts.contains((this, e))) node.nexts.append((this, e))
    if (!this.prevs.contains((node, e))) this.prevs.append((node, e))
    node
  }

  def delete(node: Node[T], e: Edge = null): Node[T] = {
    if (e != null) {
      if (node.prevs.contains((this, e))) node.prevs -= ((this, e))
      if (this.nexts.contains((node, e))) this.nexts -= ((node, e))
    } else {
      node.prevs.filter(_._1 == self).foreach(k => node.prevs -= k)
      this.nexts.filter(_._1 == node).foreach(k => this.nexts -= k)
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

  def setElement(e: T): this.type = {
    element = e
    this
  }

  def graph(reverse: Boolean = false): DirectedGraph[T] = {
    new DirectedGraph[T](this, reverse)
  }
}

object Node {
  def apply[T](element: T): Node[T] = new Node(element)
}

private[sparkdl] class Edge(val fromIndex: Option[Int]) extends Serializable {
  override def toString: String = {
    s"Edge(fromIndex: $fromIndex)"
  }

  def newInstance(): Edge = {
    fromIndex match {
      case Some(index) => Edge(index)
      case None => Edge()
    }
  }
}

object Edge {
  def apply(value: Int): Edge = new Edge(Some(value))

  def apply(): Edge = new Edge(None)
}
