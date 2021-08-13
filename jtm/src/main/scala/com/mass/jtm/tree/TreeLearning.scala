package com.mass.jtm.tree

import java.io.{BufferedReader, FileNotFoundException, InputStreamReader}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.util.{Failure, Success, Using}

import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.utils.{Engine, T, Table, Util, FileReader => DistFileReader}
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
    numThreads: Int,
    delimiter: String = ",") {
  import TreeLearning._

  private val itemSequenceMap = mutable.Map[Int, Array[Int]]()
  protected var tree: JTMTree = _
  protected var maxLevel: Int = -1
  private var dlModel: Array[Module[Float]] = _
  private val useMask: Boolean = if (modelName.toLowerCase() == "din") true else false

  def load(dataPath: String, treePath: String, modelPath: String): Unit = {
    readDataFile(dataPath)
    dlModel = duplicateModel(modelPath, numThreads)
    tree = JTMTree(treePath)
    // levels start from 0, end with (maxLevel - 1)
    maxLevel = tree.maxLevel - 1
  }

  def readDataFile(dataPath: String): Unit = {
    val tmpMap = mutable.Map.empty[Int, ArrayBuffer[Int]]
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

  // When num of nodes in one level is smaller than numThreads,
  // parallel computing item weights in ONE node.
  // Otherwise parallel computing item weights among nodes.
  def run(outputTreePath: String): Unit = {
    var level, _gap = gap
    val projectionPi = mutable.Map[Int, Int]()
    for (itemCode <- tree.leafCodes) {
      val itemId = tree.codeNodeMap(itemCode).id
      projectionPi(itemId) = 0  // first all assign to root node
    }

    val totalStart = System.nanoTime()
    val bufferProjections = Array.fill[mutable.Map[Int, Int]](numThreads)(mutable.Map.empty)
    while (_gap > 0) {
      val oldLevel = level - _gap
      val currentNodes = tree.getAllNodesAtLevel(oldLevel)

      val levelStart = System.nanoTime()
      if (currentNodes.length < numThreads) {
        val _projection = projectionPi.toArray
        currentNodes.foreach { node =>
          val itemsAssignedToNode = _projection.filter(_._2 == node).map(_._1)
          projectionPi ++= getChildrenProjection(
            oldLevel, level, node, itemsAssignedToNode, parallelItems = true)
        }
      } else {
        val taskSize = currentNodes.length / numThreads
        val extraSize = currentNodes.length % numThreads
        Engine.default.invokeAndWait(
          (0 until numThreads).map(i => () => {
            val subProjection =
              if (bufferProjections(i).isEmpty) {
                projectionPi.toArray
              } else {
                bufferProjections(i).toArray
              }
            val start = i * taskSize + math.min(i, extraSize)
            val end = start + taskSize + (if (i < extraSize) 1 else 0)
            (start until end).map(currentNodes(_)).foreach { node =>
              val itemsAssignedToNode = subProjection.filter(_._2 == node).map(_._1)
              bufferProjections(i) ++= getChildrenProjection(
                oldLevel, level, node, itemsAssignedToNode, modelIdx = i, parallelItems = false)
            }
          })
        )
      }
      val levelEnd = System.nanoTime()
      println(f"level $level assign time:  ${(levelEnd - levelStart) / 1e9d}%.6fs")

      _gap = math.min(_gap, maxLevel - level)
      level += _gap
    }

    bufferProjections.foldLeft(projectionPi)((a, b) => a ++= b)
    val totalEnd = System.nanoTime()
    println(f"total tree learning time: ${(totalEnd - totalStart) / 1e9d}%.6fs")
    tree.writeTree(projectionPi, outputTreePath)
  }

  protected def getChildrenProjection(
      oldLevel: Int,
      level: Int,
      node: Int,
      itemsAssignedToNode: Array[Int],
      modelIdx: Int = 0,
      parallelItems: Boolean): mutable.Map[Int, Int] = {

    val nodeItemMapWithWeights = mutable.Map[Int, ArrayBuffer[ItemInfo]]()
    val oldItemNodeMap = mutable.Map[Int, Int]()
    val maxAssignNum = math.pow(2, maxLevel - level).toInt
    val childrenAtLevel = tree.getChildrenAtLevel(node, oldLevel, level)
    val (candidateNodes, candidateWeights) = computeWeightsForItemsAtLevel(
      itemsAssignedToNode, node, childrenAtLevel, level, modelIdx, parallelItems)

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

    val childrenProjection = mutable.Map[Int, Int]()
    nodeItemMapWithWeights.foreach { case (node, items) =>
      require(items.length <= maxAssignNum, "items in one node should not exceed maxAssignNum")
      items.foreach(i => childrenProjection(i.id) = node)
    }
    childrenProjection
  }

  private def computeWeightsForItemsAtLevel(
      itemsAssignedToNode: Array[Int],
      currentNode: Int,
      childrenNodes: Array[Int],
      level: Int,
      modelIdx: Int,
      parallelItems: Boolean): (mutable.Map[Int, Array[Int]], mutable.Map[Int, Array[Float]]) = {

    val candidateNodesOfItems = mutable.Map.empty[Int, Array[Int]]
    val candidateWeightsOfItems = mutable.Map.empty[Int, Array[Float]]

    if (parallelItems) {
      val taskSize = itemsAssignedToNode.length / numThreads
      val extraSize = itemsAssignedToNode.length % numThreads
      val realParallelism = if (taskSize == 0) extraSize else numThreads
      val bufferNodes = Array.fill[mutable.Map[Int, Array[Int]]](realParallelism)(mutable.Map.empty)
      val bufferWeights = Array.fill[mutable.Map[Int, Array[Float]]](realParallelism)(mutable.Map.empty)
      Engine.default.invokeAndWait(
        (0 until realParallelism).map(i => () => {
          val start = i * taskSize + math.min(i, extraSize)
          val end = start + taskSize + (if (i < extraSize) 1 else 0)
          (start until end).map(itemsAssignedToNode(_)).foreach { item =>
            val childrenWeights = childrenNodes.map { node =>
              aggregateWeights(item, currentNode, node, level, i)
            }
            val index = childrenWeights.argSort(inplace = true).reverse
            bufferNodes(i)(item) = index.map(childrenNodes(_))
            bufferWeights(i)(item) = index.map(childrenWeights(_))
          }
        })
      )
      bufferNodes.foldLeft(candidateNodesOfItems)((a, b) => a ++= b)
      bufferWeights.foldLeft(candidateWeightsOfItems)((a, b) => a ++= b)
    } else {
      itemsAssignedToNode.foreach { item =>
        val childrenWeights = childrenNodes.map { childNode =>
          aggregateWeights(item, currentNode, childNode, level, modelIdx)
        }
        // sort according to descending weight
        val index = childrenWeights.argSort(inplace = true).reverse
        candidateNodesOfItems(item) = index.map(childrenNodes(_))
        candidateWeightsOfItems(item) = index.map(childrenWeights(_))
      }
    }

    (candidateNodesOfItems, candidateWeightsOfItems)
  }

  private def aggregateWeights(
      item: Int,
      currentNode: Int,
      childNode: Int,
      level: Int,
      modelIdx: Int): Float = {
    // items that never appeared as target are assigned low weights
    if (!itemSequenceMap.contains(item)) return -1e6f
    var weights = 0.0f
    var node = childNode
    var _level = level
    while (node > currentNode) {
      val sampleSet = buildFeatures(tree, itemSequenceMap(item), node, seqLen, _level,
        useMask, hierarchical, minLevel)
      weights += dlModel(modelIdx).forward(sampleSet).asInstanceOf[Tensor[Float]].storage().array().sum
      node = (node - 1) / 2
      _level -= 1
    }
    weights
  }

  private def rebalance(
      nodeItemMapWithWeights: mutable.Map[Int, ArrayBuffer[ItemInfo]],
      oldItemNodeMap: mutable.Map[Int, Int],
      childrenAtLevel: Array[Int],
      maxAssignNum: Int,
      candidateNodesOfItems: mutable.Map[Int, Array[Int]],
      candidateWeightsOfItems: mutable.Map[Int, Array[Float]]): Unit = {

    val processedNodes = new mutable.HashSet[Int]()
    var finished = false
    while (!finished) {
      val initValue = (0, -1)
      val (maxAssignCount, maxAssignNode) = childrenAtLevel.foldLeft(initValue) { (init, node) =>
        if (!processedNodes.contains(node)
            && nodeItemMapWithWeights.contains(node)
            && nodeItemMapWithWeights(node).length > init._1) {
          (nodeItemMapWithWeights(node).length, node)
        } else {
          init
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
      numThreads: Int = 1,
      delimiter: String = ","): TreeLearning = {
    new TreeLearning(modelName, gap, seqLen, hierarchical, minLevel, numThreads, delimiter)
  }

  private def duplicateModel(modelPath: String, num: Int): Array[Module[Float]] = {
    val loadedModel = Serialization.loadModel(modelPath)
    val weights: Array[Tensor[Float]] = Util.getAndClearWeightBias(loadedModel.parameters())
    val models: Array[Module[Float]] = (1 to num).toArray.map { _ =>
      val m = loadedModel.cloneModule()
      Util.putWeightBias(weights, m)
      m
    }
    models
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
      val masks =
        if (mask.isEmpty) {
          Tensor[Int]()
        } else {
          Tensor(mask.toArray, Array(mask.length))
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
      oldItemNodeMap: mutable.Map[Int, Int]): ArrayBuffer[ItemInfo] = {

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
