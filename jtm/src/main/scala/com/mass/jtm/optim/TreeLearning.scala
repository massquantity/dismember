package com.mass.jtm.optim

import java.io.{BufferedReader, InputStreamReader}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Using

import com.mass.jtm.tree.{JTMTree, TreeUtil}
import com.mass.scalann.tensor.Tensor
import com.mass.scalann.utils.{Engine, Table, FileReader => DistFileReader}
import com.mass.scalann.Module

trait TreeLearning {
  import TreeLearning._

  val dataPath: String
  val treePath: String
  val modelPath: String
  val gap: Int
  val seqLen: Int
  val hierarchical: Boolean
  val minLevel: Int
  val numThreads: Int
  val useMask: Boolean

  private val itemSequenceMap: Map[Int, Array[Int]] = readDataFile(dataPath)
  private val dlModel: Array[Module[Float]] = TreeUtil.duplicateModel(modelPath, numThreads)
  protected[jtm] val tree: JTMTree = JTMTree(treePath)
  lazy val maxLevel: Int = tree.maxLevel

  def optimize(): Map[Int, Int]

  def readDataFile(dataPath: String): Map[Int, Array[Int]] = {
    val fileReader = DistFileReader(dataPath)
    val inputStream = fileReader.open()
    val lines = Using.resource(new BufferedReader(new InputStreamReader(inputStream))) { input =>
      Iterator.continually(input.readLine()).takeWhile(_ != null).toSeq
    }
    lines
      .map(_.trim.split(","))
      .groupMapReduce(_.last.toInt)(_.drop(1).dropRight(1).map(_.toInt).toVector)(_ ++ _)
      .view
      .mapValues(_.toArray)
      .toMap
  }

  protected def getChildrenProjection(
    oldLevel: Int,
    level: Int,
    node: Int,
    itemsAssignedToNode: Array[Int],
    modelIdx: Int = 0,
    parallelItems: Boolean
  ): Map[Int, Int] = {
    val maxAssignNum = math.pow(2, maxLevel - level).toInt
    val childrenAtLevel = tree.getChildrenAtLevel(node, oldLevel, level)
    val candidateNodeWeights = computeWeightsForItemsAtLevel(
      itemsAssignedToNode,
      node,
      childrenAtLevel,
      level,
      modelIdx,
      parallelItems
    )
    val nodeItemMapWithWeights = itemsAssignedToNode.map { item =>
      val (maxWeightChildNode, maxWeight) = candidateNodeWeights(item).head
      maxWeightChildNode -> ItemInfo(item, maxWeight, 1)
    }.groupMap(_._1)(_._2)

    val oldItemNodeMap = itemsAssignedToNode.map { item =>
      item -> tree.getAncestorAtLevel(item, level)
    }.toMap

    val balancedNodesMap = reBalance(
      nodeItemMapWithWeights,
      oldItemNodeMap,
      childrenAtLevel,
      maxAssignNum,
      candidateNodeWeights
    )
    balancedNodesMap.foreach { case (_, items) =>
      assert(items.length <= maxAssignNum,
        s"items in one node should not exceed maxAssignNum, " +
          s"items length: ${items.length}, " +
          s"maxAssignNum: $maxAssignNum")
    }

    for {
      (node, items) <- balancedNodesMap
      i <- items
    } yield i.id -> node
  }

  private def computeWeightsForItemsAtLevel(
    itemsAssignedToNode: Array[Int],
    currentNode: Int,
    childrenNodes: Array[Int],
    level: Int,
    modelIdx: Int,
    parallelItems: Boolean
  ): Map[Int, Array[(Int, Float)]] = {
    if (parallelItems) {
      val taskSize = itemsAssignedToNode.length / numThreads
      val extraSize = itemsAssignedToNode.length % numThreads
      val realParallelism = if (taskSize == 0) extraSize else numThreads
      Engine.default.invokeAndWait(
        (0 until realParallelism).map { i => () =>
          val start = i * taskSize + math.min(i, extraSize)
          val end = start + taskSize + (if (i < extraSize) 1 else 0)
          sortNodeWeights(
            itemsAssignedToNode.slice(start, end),
            currentNode,
            childrenNodes,
            level,
            i
          )
        }
      ).reduce(_ ++ _)
    } else {
      sortNodeWeights(
        itemsAssignedToNode,
        currentNode,
        childrenNodes,
        level,
        modelIdx
      )
    }
  }

  private def sortNodeWeights(
    itemsAssignedToNode: Array[Int],
    currentNode: Int,
    childrenNodes: Array[Int],
    level: Int,
    modelIdx: Int,
  ): Map[Int, Array[(Int, Float)]] = {
    itemsAssignedToNode.map { item =>
      val childrenWeights = childrenNodes.map { childNode =>
        aggregateWeights(item, currentNode, childNode, level, modelIdx)
      }
      item -> childrenNodes.zip(childrenWeights).sortBy(_._2)(Ordering[Float].reverse)
    }.toMap
  }

  private def aggregateWeights(
    item: Int,
    currentNode: Int,
    childNode: Int,
    level: Int,
    modelIdx: Int
  ): Float = {
    // items that never appeared as target are assigned low weights
    if (!itemSequenceMap.contains(item)) return -1e6f
    var weights = 0.0f
    var node = childNode
    val itemSeq = itemSequenceMap(item)
    var _level = level
    while (node > currentNode) {
      val sampleSet = buildFeatures(itemSeq, node, _level)
      // use Tensor sum
      val score = dlModel(modelIdx).forward(sampleSet).toTensor[Float].sum()
      weights += score
      node = (node - 1) / 2
      _level -= 1
    }
    weights
  }

  private def buildFeatures(sequence: Array[Int], node: Int, level: Int): Table = {
    val length = sequence.length / seqLen
    val targetItems = Tensor(Array.fill[Int](length)(node), Array(length, 1))
    if (useMask) {
      val (seqCodes, mask) = tree.idToCodeWithMask(sequence, level, hierarchical, minLevel)
      val seqItems = Tensor(seqCodes, Array(length, seqLen))
      val masks =
        if (mask.isEmpty) {
          Tensor[Int]()
        } else {
          Tensor(mask.toArray, Array(mask.length))
        }
      Table(targetItems, seqItems, masks)
    } else {
      val seqCodes = tree.idToCode(sequence, level, hierarchical, minLevel)
      val seqItems = Tensor(seqCodes, Array(length, seqLen))
      Table(targetItems, seqItems)
    }
  }
}

object TreeLearning {

  case class ItemInfo(id: Int, weight: Float, nextWeightIdx: Int)

  private def getMaxNode(
    nodeItemMap: mutable.Map[Int, ArrayBuffer[ItemInfo]],
    nodes: Array[Int],
    processedNodes: mutable.HashSet[Int]
  ): (Int, Int) = {
    nodes.map { node =>
      if (!processedNodes.contains(node) && nodeItemMap.contains(node)) {
        (nodeItemMap(node).length, node)
      } else {
        (-1, 0)
      }
    }.maxBy(_._1)
  }

  def reBalance(
    nodeItemMapWithWeights: Map[Int, Array[ItemInfo]],
    oldItemNodeMap: Map[Int, Int],
    childrenAtLevel: Array[Int],
    maxAssignNum: Int,
    candidateNodeWeightsOfItems: Map[Int, Array[(Int, Float)]]
  ): Map[Int, ArrayBuffer[ItemInfo]] = {
    implicit val ord: Ordering[(Boolean, Float)] = Ordering.Tuple2(Ordering.Boolean, Ordering[Float].reverse)
    val resMap = nodeItemMapWithWeights.view.mapValues(_.to(ArrayBuffer)).to(mutable.Map)
    val processedNodes = new mutable.HashSet[Int]()
    var finished = false
    while (!finished) {
      val (maxAssignCount, maxAssignNode) = getMaxNode(
        resMap,
        childrenAtLevel,
        processedNodes
      )
      if (maxAssignCount <= maxAssignNum) {
        finished = true
      } else {
        processedNodes += maxAssignNode
        // start from maxAssignNum, and move the redundant items to other nodes
        val (chosenItems, redundantItems) = resMap(maxAssignNode)
          .sortBy(i => (oldItemNodeMap(i.id) != maxAssignNode, i.weight))
          .splitAt(maxAssignNum)
        resMap(maxAssignNode) = chosenItems
        redundantItems.foreach { i =>
          val candidateNodeWeights = candidateNodeWeightsOfItems(i.id)
          var index = i.nextWeightIdx
          var found = false
          while (!found && index < candidateNodeWeights.length) {
            val (node, weight) = candidateNodeWeights(index)
            if (!processedNodes.contains(node)) {
              found = true
              if (resMap.contains(node)) {
                // set index to next max weight
                resMap(node) += ItemInfo(i.id, weight, index + 1)
              } else {
                resMap(node) = ArrayBuffer(ItemInfo(i.id, weight, index + 1))
              }
            }
            index += 1
          }
        }
      }
    }
    resMap.toMap
  }
}
