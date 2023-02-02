package com.mass.otm.tree

import java.io.{BufferedReader, FileReader}

import scala.annotation.tailrec
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Using

import com.mass.otm.{paddingIdx, upperLog2, DeepModel}
import com.mass.otm.model.ModelUtil._
import com.mass.scalann.tensor.Tensor
import com.mass.scalann.tensor.TensorNumeric.NumericDouble
import com.mass.scalann.utils.{Engine, Table}
import com.mass.tdm.utils.Serialization
import org.apache.log4j.Logger

class TreeConstruction(
    dataPath: String,
    modelPath: String,
    mappingPath: String,
    gap: Int,
    labelNum: Int,
    minSeqLen: Int,
    seqLen: Int,
    splitRatio: Double,
    numThreads: Int,
    useMask: Boolean
) {
  import TreeConstruction._
  val logger: Logger = Logger.getLogger(getClass)

  val itemIdMapping: Map[Int, Int] = Serialization.loadMapping(mappingPath)
  val leafLevel: Int = upperLog2(itemIdMapping.size)
  val clonedModels: IndexedSeq[DeepModel[Double]] = duplicateModels(modelPath, numThreads)
  val itemSequenceMap: Map[Int, Array[Int]] = readDataFile(
    dataPath,
    labelNum,
    minSeqLen,
    seqLen,
    splitRatio,
    itemIdMapping
  )

  def run(): Map[Int, Int] = {
    // first all assign to root node
    val initProjection = itemIdMapping.keys.map(_ -> 0).toMap
    (0 until leafLevel by gap).foldLeft(initProjection) { case (oldProjection, oldLevel) =>
      val levelStart = System.nanoTime()
      val level = math.min(leafLevel, oldLevel + gap)
      val reverseProjection = oldProjection.toArray.groupMap(_._2)(_._1)
      val currentNodes = getAllNodesAtLevel(oldLevel).filter(reverseProjection.contains)
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

  private def getChildrenProjection(
      oldLevel: Int,
      level: Int,
      node: Int,
      itemsAssignedToNode: Array[Int],
      modelIdx: Int = 0,
      parallelItems: Boolean
  ): Map[Int, Int] = {
    val maxAssignNum = math.pow(2, leafLevel - level).toInt
    val childrenAtLevel = getChildrenAtLevel(node, oldLevel, level)
    val candidateNodeWeights = computeWeightsForItemsAtLevel(
      itemsAssignedToNode,
      node,
      childrenAtLevel,
      modelIdx,
      parallelItems
    )
    val nodeItemMapWithWeights = itemsAssignedToNode
      .map { item =>
        val (maxWeightChildNode, maxWeight) = candidateNodeWeights(item).head
        maxWeightChildNode -> ItemInfo(item, maxWeight, 1)
      }
      .groupMap(_._1)(_._2)

    val oldItemNodeMap = itemsAssignedToNode.map { item =>
      item -> getAncestorAtLevel(item, level, itemIdMapping)
    }.toMap

    val balancedNodesMap = reBalance(
      nodeItemMapWithWeights,
      oldItemNodeMap,
      childrenAtLevel,
      maxAssignNum,
      candidateNodeWeights
    )
    balancedNodesMap.foreach { case (_, items) =>
      assert(
        items.length <= maxAssignNum,
        s"items in one node should not exceed maxAssignNum, " +
          s"items length: ${items.length}, " +
          s"maxAssignNum: $maxAssignNum"
      )
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
      modelIdx: Int,
      parallelItems: Boolean
  ): Map[Int, Array[(Int, Double)]] = {
    if (parallelItems) {
      val taskSize = itemsAssignedToNode.length / numThreads
      val extraSize = itemsAssignedToNode.length % numThreads
      val realParallelism = if (taskSize == 0) extraSize else numThreads
      Engine.default
        .invokeAndWait(
          (0 until realParallelism).map { i => () =>
            val start = i * taskSize + math.min(i, extraSize)
            val end = start + taskSize + (if (i < extraSize) 1 else 0)
            sortNodeWeights(
              itemsAssignedToNode.slice(start, end),
              currentNode,
              childrenNodes,
              i
            )
          }
        )
        .reduce(_ ++ _)
    } else {
      sortNodeWeights(
        itemsAssignedToNode,
        currentNode,
        childrenNodes,
        modelIdx
      )
    }
  }

  private def sortNodeWeights(
      itemsAssignedToNode: Array[Int],
      currentNode: Int,
      childrenNodes: Array[Int],
      modelIdx: Int
  ): Map[Int, Array[(Int, Double)]] = {
    itemsAssignedToNode.map { item =>
      val childrenWeights = childrenNodes.map { childNode =>
        aggregateWeights(item, currentNode, childNode, modelIdx)
      }
      item -> childrenNodes.zip(childrenWeights).sortBy(_._2)(Ordering[Double].reverse)
    }.toMap
  }

  private def aggregateWeights(
      item: Int,
      currentNode: Int,
      childNode: Int,
      modelIdx: Int
  ): Double = {
    // items that never appeared as target are assigned low weights
    if (!itemSequenceMap.contains(item)) return -1e6
    var weights = 0.0
    var node = childNode
    val itemSeq = itemSequenceMap(item)
    while (node > currentNode) {
      val sampleSet = buildFeatures(itemSeq, node)
      // use Tensor sum
      val score = clonedModels(modelIdx).forward(sampleSet).toTensor.sum()
      weights += score
      node = (node - 1) >> 1
    }
    weights
  }

  private def buildFeatures(sequence: Array[Int], node: Int): Table = {
    val length = sequence.length / seqLen
    val nodesTensor = Tensor(Array.fill(length)(node), Array(length, 1))
    val seqTensor = Tensor(sequence, Array(length, seqLen))
    if (useMask) {
      val masks = sequence.zipWithIndex.filter(_._1 == paddingIdx).map(_._2)
      val maskTensor = if (masks.isEmpty) {
        Tensor[Int]()
      } else {
        Tensor(masks, Array(masks.length))
      }
      Table(nodesTensor, seqTensor, maskTensor)
    } else {
      Table(nodesTensor, seqTensor)
    }
  }
}

object TreeConstruction {

  case class ItemInfo(id: Int, weight: Double, nextWeightIdx: Int)

  def apply(
      dataPath: String,
      modelPath: String,
      mappingPath: String,
      gap: Int,
      labelNum: Int,
      minSeqLen: Int,
      seqLen: Int,
      splitRatio: Double,
      numThreads: Int,
      useMask: Boolean
  ): TreeConstruction = {
    new TreeConstruction(
      dataPath,
      modelPath,
      mappingPath,
      gap,
      labelNum,
      minSeqLen,
      seqLen,
      splitRatio,
      numThreads,
      useMask
    )
  }

  def getAncestorAtLevel(item: Int, level: Int, itemIdMapping: Map[Int, Int]): Int = {
    val maxNodeAtLevel = math.pow(2, level + 1).toInt - 1
    @tailrec
    def upsert(node: Int): Int = {
      if (node < maxNodeAtLevel) {
        node
      } else {
        upsert((node - 1) >> 1)
      }
    }
    upsert(itemIdMapping(item))
  }

  def getAllNodesAtLevel(level: Int): Seq[Int] = {
    val levelStart = math.pow(2, level).toInt - 1
    val levelEnd = levelStart * 2 + 1
    Seq.range(levelStart, levelEnd)
  }

  def getChildrenAtLevel(ancestor: Int, oldLevel: Int, level: Int): Array[Int] = {
    (oldLevel until level).foldLeft(Array(ancestor)) { case (nodes, _) =>
      nodes.flatMap(n => Array(n * 2 + 1, n * 2 + 2))
    }
  }

  def getMaxNode(
      nodeItemMap: mutable.Map[Int, ArrayBuffer[ItemInfo]],
      nodes: Array[Int],
      processedNodes: mutable.HashSet[Int]
  ): (Int, Int) = {
    nodes
      .map { node =>
        if (!processedNodes.contains(node) && nodeItemMap.contains(node)) {
          (nodeItemMap(node).length, node)
        } else {
          (-1, 0)
        }
      }
      .maxBy(_._1)
  }

  def reBalance(
      nodeItemMapWithWeights: Map[Int, Array[ItemInfo]],
      oldItemNodeMap: Map[Int, Int],
      childrenAtLevel: Array[Int],
      maxAssignNum: Int,
      candidateNodeWeightsOfItems: Map[Int, Array[(Int, Double)]]
  ): Map[Int, ArrayBuffer[ItemInfo]] = {
    implicit val ord: Ordering[(Boolean, Double)] =
      Ordering.Tuple2(Ordering.Boolean, Ordering[Double].reverse)
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

  def readDataFile(dataPath: String, itemIdMapping: Map[Int, Int]): Map[Int, Array[Int]] = {
    val lines = Using.resource(new BufferedReader(new FileReader(dataPath))) { input =>
      Iterator.continually(input.readLine()).takeWhile(_ != null).toSeq
    }
    lines
      .map(_.trim.split(","))
      .groupMapReduce(_.last.toInt)(_.drop(1).dropRight(1).map(_.toInt).toVector)(_ ++ _)
      .view
      .mapValues(_.toArray.map(itemIdMapping.getOrElse(_, paddingIdx)))
      .toMap
  }

  def readDataFile(
      dataPath: String,
      labelNum: Int,
      minSeqLen: Int,
      seqLen: Int,
      splitRatio: Double,
      itemIdMapping: Map[Int, Int]
  ): Map[Int, Array[Int]] = {
    case class InitSample(user: Int, item: Int, timestamp: Long)
    val data = Using.resource(new BufferedReader(new FileReader(dataPath))) { input =>
      Iterator
        .continually(input.readLine())
        .drop(1)
        .takeWhile(_ != null)
        .map { line =>
          val s = line.trim.split(",")
          InitSample(s(0).toInt, s(1).toInt, s(2).toLong)
        }
        .toVector
    }
    val allItemSeqs = data
      .groupBy(_.user)
      .map { case (_, samples) =>
        samples.sortBy(_.timestamp).map(_.item).distinct.map(itemIdMapping(_))
      }
      .filter(_.length >= minSeqLen + labelNum)

    val idItemMapping = itemIdMapping.map(_.swap)
    val paddingSeq = Seq.fill(seqLen - minSeqLen)(paddingIdx)
    val seqTargetPairs = allItemSeqs.flatMap { items =>
      if (items.length == minSeqLen + labelNum) {
        val fullSeq = paddingSeq ++: items.take(minSeqLen)
        val labels = items.drop(minSeqLen)
        labels.map(fullSeq -> idItemMapping(_))
      } else {
        val fullSeq = paddingSeq ++: items
        val splitPoint = math.ceil((items.length - minSeqLen) * splitRatio).toInt
        val trainData = fullSeq.take(splitPoint + seqLen)
        for {
          trainSeq <- trainData.sliding(seqLen + labelNum)
          (seq, labels) = trainSeq.splitAt(seqLen)
          target <- labels
        } yield seq -> idItemMapping(target)
      }
    }
    seqTargetPairs
      .groupMapReduce(_._2)(_._1)(_ ++ _)
      .view
      .mapValues(_.toArray)
      .toMap
  }

  def duplicateModels(
      modelPath: String,
      numThread: Int
  ): IndexedSeq[DeepModel[Double]] = {
    val model = Serialization.loadModel[Double](modelPath)
    compactParameters(model)
    val modelWeights = extractWeights(model)
    clearParameters(model)
    val clonedModels = (1 to numThread).map { _ =>
      val m = model.cloneModule()
      putWeights(m, modelWeights)
      initGradients(m, modelWeights)
      m
    }
    putWeights(model, modelWeights)
    initGradients(model, modelWeights)
    clonedModels
  }
}
