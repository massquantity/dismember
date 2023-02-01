package com.mass.jtm.optim

import com.mass.scalann.utils.Engine
import org.apache.log4j.Logger

class JTM(
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
  val logger: Logger = Logger.getLogger(getClass)

  // When num of nodes in one level is smaller than numThreads,
  // parallel computing item weights in ONE node.
  // Otherwise parallel computing item weights among nodes.
  override def optimize(): Map[Int, Int] = {
    val initProjection = tree.leafCodes.map { itemCode =>
      val itemId = tree.codeNodeMap(itemCode).id
      itemId -> 0 // first all assign to root node
    }.toMap

    (0 until maxLevel by gap).foldLeft(initProjection) { case (oldProjection, oldLevel) =>
      val levelStart = System.nanoTime()
      val level = math.min(maxLevel, oldLevel + gap)
      val reverseProjection = oldProjection.toArray.groupMap(_._2)(_._1)
      val currentNodes = tree.getAllNodesAtLevel(oldLevel).filter(reverseProjection.contains)
      val newProjections =
        if (currentNodes.length < numThreads) {
          currentNodes.map { node =>
            val itemsAssignedToNode = reverseProjection(node)
            getChildrenProjection(
              oldLevel,
              level,
              node,
              itemsAssignedToNode,
              parallelItems = true
            )
          }
        } else {
          val taskSize = currentNodes.length / numThreads
          val extraSize = currentNodes.length % numThreads
          Engine.default.invokeAndWait(
            (0 until numThreads).map { i => () =>
              val start = i * taskSize + math.min(i, extraSize)
              val end = start + taskSize + (if (i < extraSize) 1 else 0)
              currentNodes
                .slice(start, end)
                .flatMap { node =>
                  val itemsAssignedToNode = reverseProjection(node)
                  getChildrenProjection(
                    oldLevel,
                    level,
                    node,
                    itemsAssignedToNode,
                    modelIdx = i,
                    parallelItems = false
                  )
                }
                .toMap
            }
          )
        }
      val levelEnd = System.nanoTime()
      logger.info(f"level $level assign time:  ${(levelEnd - levelStart) / 1e9d}%.6fs")
      newProjections.foldLeft(oldProjection)(_ ++ _)
    }
  }
}

object JTM {

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
  ): JTM = {
    new JTM(
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
}
