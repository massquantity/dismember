package com.mass.jtm.tree

import scala.collection.mutable

import com.mass.sparkdl.utils.Engine

class TreeLearningAsync(
    modelName: String,
    gap: Int,
    seqLen: Int,
    hierarchical: Boolean,
    minLevel: Int,
    numThreads: Int,
    delimiter: String = ",") extends TreeLearning(
      modelName, gap, seqLen, hierarchical, minLevel, numThreads, delimiter) {
  import TreeLearningAsync._

  require(gap > 0, s"gap must be positive, but got $gap")
  require(isPowerOf2(numThreads), s"numThreads should be power of 2, but got $numThreads")

  override def run(outputTreePath: String): Unit = {
    var level, _gap = gap
    val projectionPi = mutable.Map[Int, Int]()
    for (itemCode <- tree.leafCodes) {
      val itemId = tree.codeNodeMap(itemCode).id
      projectionPi(itemId) = 0
    }

    val totalStart = System.nanoTime()
    var asyncParallel = false
    var asyncStartLevel = 0
    while (_gap > 0 && !asyncParallel) {
      val levelStart = System.nanoTime()
      val oldLevel = level - _gap
      val currentNodes = tree.getAllNodesAtLevel(oldLevel)
      val childrenNodes = tree.getAllNodesAtLevel(level)
      if (childrenNodes.length >= numThreads) {
        asyncParallel = true
        asyncStartLevel = level
      }

      val _projection = projectionPi.toArray
      currentNodes.foreach { node =>
        val itemAssignedToNode = _projection.filter(_._2 == node).map(_._1)
        projectionPi ++= getChildrenProjection(
          oldLevel, level, node, itemAssignedToNode, parallelItems = true)
      }

      val levelEnd = System.nanoTime()
      println(f"level $level assign time:  ${(levelEnd - levelStart) / 1e9d}%.6fs")
      _gap = math.min(_gap, maxLevel - level)
      level += _gap
    }

    if (asyncParallel) {
      println("asynchronous learning begin...")
      val asyncStartNodes = tree.getAllNodesAtLevel(asyncStartLevel)
      Engine.default.invokeAndWait(
        (0 until numThreads).map(i => () => {
          singlePathAssign(asyncStartNodes(i), level, _gap, i, projectionPi)
        })
      ).foreach(projectionPi ++= _)
    }

    val totalEnd = System.nanoTime()
    println(f"total tree learning time: ${(totalEnd - totalStart) / 1e9d}%.6fs")
    tree.writeTree(projectionPi, outputTreePath)
  }

  def singlePathAssign(
      initNode: Int,
      level: Int,
      gap: Int,
      modelIdx: Int,
      initProjection: mutable.Map[Int, Int]): mutable.Map[Int, Int] = {

    val start = System.nanoTime()
    val subProjection = mutable.Map[Int, Int]()
    val initialLevel = level - gap
    var (_level, _gap) = (level, gap)
    while (_gap > 0) {
      val oldLevel = _level - _gap
      val (currentNodes, _projection) =
        if (subProjection.isEmpty) {
          (Array(initNode), initProjection.toArray)
        } else {
          (tree.getChildrenAtLevel(initNode, initialLevel, oldLevel), subProjection.toArray)
        }

      currentNodes.foreach { node =>
        val itemAssignedToNode = _projection.filter(_._2 == node).map(_._1)
        subProjection ++= getChildrenProjection(
          oldLevel, _level, node, itemAssignedToNode, modelIdx, parallelItems = false)
      }
      _gap = math.min(_gap, maxLevel - _level)
      _level += _gap
    }
    val end = System.nanoTime()
    println(f"thread $modelIdx assign time:  ${(end - start) / 1e9d}%.6fs")
    subProjection
  }
}


object TreeLearningAsync {

  def apply(
      modelName: String,
      gap: Int,
      seqLen: Int,
      hierarchical: Boolean,
      minLevel: Int,
      numThreads: Int = 1,
      delimiter: String = ","): TreeLearning = {
    new TreeLearningAsync(modelName, gap, seqLen, hierarchical, minLevel, numThreads, delimiter)
  }

  private def isPowerOf2(n: Int): Boolean = {
    n > 0 && (n & (n - 1)) == 0
  }
}
