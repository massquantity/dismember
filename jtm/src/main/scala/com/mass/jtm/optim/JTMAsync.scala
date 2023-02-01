package com.mass.jtm.optim

import scala.collection.mutable

import com.mass.scalann.utils.Engine
import org.apache.log4j.Logger

class JTMAsync(
    override val dataPath: String,
    override val treePath: String,
    override val modelPath: String,
    override val gap: Int,
    override val seqLen: Int,
    override val hierarchical: Boolean,
    override val minLevel: Int,
    override val numThreads: Int,
    override val useMask: Boolean
) extends TreeLearning {
  import JTMAsync._
  require(gap > 0, s"gap must be positive, but got $gap")
  require(isPowerOf2(numThreads), s"numThreads should be power of 2, but got $numThreads")
  val logger: Logger = Logger.getLogger(getClass)

  override def optimize(): Map[Int, Int] = {
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
          oldLevel,
          level,
          node,
          itemAssignedToNode,
          parallelItems = true
        )
      }

      val levelEnd = System.nanoTime()
      logger.info(f"level $level assign time:  ${(levelEnd - levelStart) / 1e9d}%.6fs")
      _gap = math.min(_gap, maxLevel - level)
      level += _gap
    }

    if (asyncParallel) {
      logger.info("asynchronous learning begin...")
      val asyncStartNodes = tree.getAllNodesAtLevel(asyncStartLevel)
      asyncStartNodes.sliding(numThreads, numThreads).foreach { nodes =>
        Engine.default
          .invokeAndWait(
            nodes.zipWithIndex.map { i => () =>
              singlePathAssign(i._1, level, _gap, i._2, projectionPi)
            }
          )
          .foreach(projectionPi ++= _)
      }
    }

    val totalEnd = System.nanoTime()
    logger.info(f"total tree learning time: ${(totalEnd - totalStart) / 1e9d}%.6fs")
    Map.empty ++ projectionPi
  }

  def singlePathAssign(
      initNode: Int,
      level: Int,
      gap: Int,
      modelIdx: Int,
      initProjection: mutable.Map[Int, Int]
  ): mutable.Map[Int, Int] = {
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
          oldLevel,
          _level,
          node,
          itemAssignedToNode,
          modelIdx,
          parallelItems = false
        )
      }
      _gap = math.min(_gap, maxLevel - _level)
      _level += _gap
    }
    val end = System.nanoTime()
    logger.info(f"thread $modelIdx assign time:  ${(end - start) / 1e9d}%.6fs")
    subProjection
  }
}

object JTMAsync {

  def apply(
      dataPath: String,
      treePath: String,
      modelPath: String,
      gap: Int,
      seqLen: Int,
      hierarchical: Boolean,
      minLevel: Int,
      numThreads: Int,
      useMask: Boolean
  ): JTMAsync = {
    new JTMAsync(
      dataPath,
      treePath,
      modelPath,
      gap,
      seqLen,
      hierarchical,
      minLevel,
      numThreads,
      useMask
    )
  }

  private def isPowerOf2(n: Int): Boolean = {
    n > 1 && (n & (n - 1)) == 0
  }
}
