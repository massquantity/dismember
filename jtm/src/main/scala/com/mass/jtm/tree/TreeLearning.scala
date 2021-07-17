package com.mass.jtm.tree

import java.io.{BufferedReader, FileNotFoundException, InputStreamReader}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.util.{Failure, Success, Using}

import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.utils.{T, Table, FileReader => DistFileReader}
import com.mass.sparkdl.Module
import com.mass.tdm.ArrayExtension
import com.mass.tdm.utils.Serialization
import spire.algebra.Order
import spire.math.{Sorting => SpireSorting}

class TreeLearning(
    modelName: String,
    gap: Int,
    seqLen: Int,
    hierarchical: Boolean,
    minLevel: Int,
    parallel: Boolean,
    numThreads: Int,
    delimiter: String = ",") {
  import TreeLearning.{buildFeatures, ItemInfo}

  private val itemSequenceMap = new mutable.HashMap[Int, Array[Int]]()
  private var tree: JTMTree = _
  private var maxLevel: Int = -1
  private var dlModel: Module[Float] = _
  private val useMask: Boolean = if (modelName.toLowerCase() == "din") true else false

  def load(dataPath: String, treePath: String, modelPath: String): Unit = {
    readDataFile(dataPath)
    dlModel = Serialization.loadModel(modelPath)
    tree = JTMTree(treePath)
    // levels start from 0, end with (maxLevel - 1)
    maxLevel = tree.maxLevel - 1
  }

  def readDataFile(dataPath: String): Unit = {
    val tmpMap = mutable.HashMap.empty[Int, ArrayBuffer[Int]]
    val fileReader = DistFileReader(dataPath)
    val inputStream = fileReader.open()

    Using(new BufferedReader(new InputStreamReader(inputStream))) { input =>
      Iterator.continually(input.readLine()).takeWhile(_ != null).foreach { line =>
        val arr = line.trim.split(delimiter)
        val seqItems = arr.slice(1, arr.length - 1).map(_.toInt)
        val targetItem = arr.last.toInt
        if (tmpMap.contains(targetItem)) {
          tmpMap(targetItem) ++= seqItems
        } else {
          tmpMap(targetItem) = ArrayBuffer(seqItems: _*)
        }
      }
    } match {
      case Success(_) =>
        inputStream.close()
        fileReader.close()
      case Failure(e: FileNotFoundException) =>
        println(s"""file "$dataPath" not found""")
        throw e
      case Failure(t: Throwable) =>
        throw t
    }

    tmpMap.foreach { case (targetItem, seqItems) =>
      itemSequenceMap(targetItem) = seqItems.toArray
    }
  }

  def run(outputTreePath: String): Unit = {
    var level, _gap = gap
    val projectionPi = new mutable.HashMap[Int, Int]()
    for (itemCode <- tree.leafCodes) {
      val itemId = tree.codeNodeMap(itemCode).id
      projectionPi(itemId) = 0  // first all assign to root node
    }

    while (_gap > 0) {
      val oldLevel = level - _gap
      val nodes = tree.getAllNodesAtLevel(oldLevel)
      val start = System.nanoTime()
      nodes.foreach { node =>
        val itemsAssignedToNode = projectionPi.toArray.filter(_._2 == node).map(_._1)
        updateProjection(projectionPi, oldLevel, level, node, itemsAssignedToNode)
      }
      val end = System.nanoTime()
      println(f"level $level time:  ${(end - start) / 1e9d}%.6fs")

      _gap = math.min(_gap, maxLevel - level)
      level += _gap
    }

    tree.writeTree(projectionPi, outputTreePath)
  }

  def updateProjection(
      projection: mutable.HashMap[Int, Int],
      oldLevel: Int,
      level: Int,
      node: Int,
      itemsAssignedToNode: Array[Int]): Unit = {

    val nodeItemMapWithWeights = new mutable.HashMap[Int, ArrayBuffer[ItemInfo]]()
    val oldItemNodeMap = new mutable.HashMap[Int, Int]()
    val maxAssignNum = math.pow(2, maxLevel - level).toInt
    val childrenAtLevel = tree.getChildrenAtLevel(node, oldLevel, level)
    val (candidateNodes, candidateWeights) = computeWeightsForItemsAtLevel(
      itemsAssignedToNode, node, childrenAtLevel, level)

    itemsAssignedToNode.foreach { item =>
      val maxWeightChildNode = candidateNodes(item).head
      val maxWeight = candidateWeights(item).head
      if (nodeItemMapWithWeights.contains(maxWeightChildNode)) {
        nodeItemMapWithWeights(maxWeightChildNode) += ItemInfo(item, maxWeight, 1)
      } else {
        nodeItemMapWithWeights(maxWeightChildNode) = ArrayBuffer(ItemInfo(item, maxWeight, 1))
      }

      oldItemNodeMap(item) = tree.getAncestorAtLevel(item, level)
    }

    rebalance(nodeItemMapWithWeights, oldItemNodeMap, childrenAtLevel, maxAssignNum,
      candidateNodes, candidateWeights)

    nodeItemMapWithWeights.foreach { case (node, items) =>
      require(items.length <= maxAssignNum)
      items.foreach(i => projection(i.id) = node)
    }
  }

  def computeWeightsForItemsAtLevel(
      itemsAssignedToNode: Array[Int],
      currentNode: Int,
      childrenNodes: Array[Int],
      level: Int): (mutable.HashMap[Int, Array[Int]], mutable.HashMap[Int, Array[Float]]) = {

    val candidateNodesOfItems = mutable.HashMap.empty[Int, Array[Int]]
    val candidateWeightsOfItems = mutable.HashMap.empty[Int, Array[Float]]

    itemsAssignedToNode.foreach { item =>
      val candidateNode = ArrayBuffer.empty[Int]
      val candidateWeight = ArrayBuffer.empty[Float]
      var i = 0
      while (i < childrenNodes.length) {
        val childNode = childrenNodes(i)
        val childWeight = aggregateWeights(item, currentNode, childNode, level)
        candidateNode += childNode
        candidateWeight += childWeight
        i += 1
      }

      // sort according to descending weight
      val index = candidateWeight.toArray.argSort(inplace = true).reverse
      candidateNodesOfItems(item) = index.map(candidateNode(_))
      candidateWeightsOfItems(item) = index.map(candidateWeight(_))
    }
    (candidateNodesOfItems, candidateWeightsOfItems)
  }

  private def aggregateWeights(item: Int, currentNode: Int, childNode: Int, level: Int): Float = {
    // items that never appeared as target are assigned low weights
    if (!itemSequenceMap.contains(item)) return -1e6f
    var weights = 0.0f
    var node = childNode
    var _level = level
    while (node > currentNode) {
      val sampleSet = buildFeatures(tree, itemSequenceMap(item), node, seqLen, _level,
        useMask, hierarchical, minLevel)
      weights += dlModel.forward(sampleSet).asInstanceOf[Tensor[Float]].storage().array().sum
      node = (node - 1) / 2
      _level -= 1
    }
    weights
  }

  private def rebalance(
      nodeItemMapWithWeights: mutable.HashMap[Int, ArrayBuffer[ItemInfo]],
      oldItemNodeMap: mutable.HashMap[Int, Int],
      childrenAtLevel: Array[Int],
      maxAssignNum: Int,
      candidateNodesOfItems: mutable.HashMap[Int, Array[Int]],
      candidateWeightsOfItems: mutable.HashMap[Int, Array[Float]]): Unit = {

    val processedNodes = new mutable.HashSet[Int]()
    var finished = false
    while (!finished) {
      var maxAssignCount = 0
      var maxAssignNode = -1
      childrenAtLevel.foreach { node =>
        if (!processedNodes.contains(node)
            && nodeItemMapWithWeights.contains(node)
            && nodeItemMapWithWeights(node).length > maxAssignCount) {
          maxAssignCount = nodeItemMapWithWeights(node).length
          maxAssignNode = node
        }
      }

      if (maxAssignCount <= maxAssignNum) {
        finished = true
      } else {
        processedNodes.add(maxAssignNode)
        val sortedItemsAndWeights = nodeItemMapWithWeights(maxAssignNode)
          .sortBy(i => (oldItemNodeMap(i.id) != maxAssignNode, i.weight))(
            Ordering.Tuple2(Ordering.Boolean, Ordering.Float.reverse))

        // start from maxAssignNum, and move the redundant items to other nodes
        var i = maxAssignNum
        while (i < sortedItemsAndWeights.length) {
          val redundantItem = sortedItemsAndWeights(i).id
          val candidateNodes = candidateNodesOfItems(redundantItem)
          val candidateWeights = candidateWeightsOfItems(redundantItem)
          var index = sortedItemsAndWeights(i).nextWeightIdx
          var found = false
          while (!found && index < candidateNodes.length) {
            val node = candidateNodes(index)
            val weight = candidateWeights(index)
            if (!processedNodes.contains(node)) {
              found = true
              if (nodeItemMapWithWeights.contains(node)) {
                // set index to next max weight
                nodeItemMapWithWeights(node) += ItemInfo(redundantItem, weight, index + 1)
              } else {
                nodeItemMapWithWeights(node) = ArrayBuffer(ItemInfo(redundantItem, weight, index + 1))
              }
            }
            index += 1
          }
          i += 1
        }

        sortedItemsAndWeights.reduceToSize(maxAssignNum)
        nodeItemMapWithWeights(maxAssignNode) = sortedItemsAndWeights
      }
    }
  }
}

object TreeLearning {

  case class ItemInfo(id: Int, weight: Float, nextWeightIdx: Int)

  def apply(
      modelName: String,
      gap: Int,
      seqLen: Int,
      hierarchical: Boolean,
      minLevel: Int,
      parallel: Boolean = false,
      numThreads: Int = 1,
      delimiter: String = ","): TreeLearning = {
    new TreeLearning(modelName, gap, seqLen, hierarchical, minLevel,
      parallel, numThreads, delimiter)
  }

  private def buildFeatures(
      tree: JTMTree,
      sequence: Array[Int],
      node: Int,
      seqLen: Int,
      level: Int,
      useMask: Boolean,
      hierarchical: Boolean,
      minLevel: Int): Table = {

    val length = sequence.length / seqLen
    val targetItems = Tensor(Array.fill[Int](length)(node), Array(length, 1))
    if (useMask) {
      val (seqCodes, mask) = tree.idToCodeWithMask(sequence, level, hierarchical, minLevel)
      val seqItems = Tensor(seqCodes, Array(length, seqLen))
      val masks = {
        if (mask.isEmpty) {
          Tensor[Int]()
        } else {
          Tensor(mask.toArray, Array(mask.length))
        }
      }
      T(targetItems, seqItems, masks)

    } else {
      val seqCodes = tree.idToCode(sequence, level, hierarchical, minLevel)
      val seqItems = Tensor(seqCodes, Array(length, seqLen))
      T(targetItems, seqItems)
    }
  }

  def sortOnFormerAndWeights(
      items: Array[ItemInfo],
      maxAssignNode: Int,
      oldItemNodeMap: mutable.HashMap[Int, Int]): ArrayBuffer[ItemInfo] = {

    SpireSorting.quickSort(items)(new Order[ItemInfo] {
      override def compare(x: ItemInfo, y: ItemInfo): Int = {
        val xNode = oldItemNodeMap(x.id)
        val yNode = oldItemNodeMap(y.id)
        if (xNode == maxAssignNode && yNode != maxAssignNode) {
          -1
        } else if (xNode != maxAssignNode && yNode == maxAssignNode) {
          1
        } else {
          java.lang.Double.compare(y.weight, x.weight)
        }
      }
    }, ClassTag[ItemInfo](getClass))

    ArrayBuffer(items: _*)
  }
}
