package com.mass.scalann.nn.graphnn

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
      val startNodes = inDegrees.view.filterKeys(inDegrees(_) == 0).keySet.toArray
      require(startNodes.nonEmpty, "There exists a cycle in the graph")
      result ++= startNodes
      startNodes.foreach { sn =>
        val nextNodes = if (!reverse) sn.nextNodes else sn.prevNodes
        nextNodes.foreach(inDegrees(_) -= 1)
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
        val nodesSet = mutable.LinkedHashSet.from(nextNodes)
        for (n <- nodesSet if !visited.contains(n) && !stack.contains(n)) {
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
        val nodesSet = mutable.LinkedHashSet.from(nextNodes)
        for (n <- nodesSet if !visited.contains(n) && !queue.contains(n)) {
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
