package com.mass.jtm.tree

import scala.collection.mutable

import com.mass.jtm.optim.TreeLearning
import com.mass.scalann.utils.Util
import com.mass.scalann.Module
import com.mass.scalann.tensor.Tensor
import com.mass.tdm.utils.Serialization

object TreeUtil {

  case class TreeMeta(leafNum: Int, maxLevel: Int, itemIds: Seq[Int])

  val getTreeMeta: TreeLearning => TreeMeta = jtmModel =>
    TreeMeta(jtmModel.tree.leafCodes.size, jtmModel.tree.maxLevel, jtmModel.tree.idCodeMap.keys.toSeq)

  def writeTree(jtmModel: TreeLearning, projectionPi: Map[Int, Int], outputTreePath: String): Unit = {
    jtmModel.tree.writeTree(projectionPi, outputTreePath)
  }

  private[jtm] def duplicateModel(modelPath: String, num: Int): Array[Module[Float]] = {
    val loadedModel = Serialization.loadModel[Float](modelPath)
    val weights: Array[Tensor[Float]] = Util.getAndClearWeightBias(loadedModel.parameters())
    val models: Array[Module[Float]] = (1 to num).toArray.map { _ =>
      val m = loadedModel.cloneModule()
      Util.putWeightBias(weights, m)
      m
    }
    models
  }

  def flattenLeaves(
    projectionPi: mutable.Map[Int, Int],
    leafCodes: mutable.BitSet,
    idCodeMap: Map[Int, Int],
    maxLevel: Int
  ): Unit = {
    val minLeafCode = math.pow(2, maxLevel).toInt - 1
    val projection = projectionPi.toArray
    val projectionLeafCodes = projection.map(_._2).filter(_ >= minLeafCode).toSet
    val unAssignedLeafCodes = leafCodes.diff(projectionLeafCodes)
    val (noPlaceItems, originalPlaceItems) = projection
      .filter(_._2 < minLeafCode)
      .partition(i => {
        val oldCode = idCodeMap(i._1)
        projectionLeafCodes.contains(oldCode)
      })

    println(s"unassigned nodes: ${unAssignedLeafCodes.size}, " +
      s"no place items: ${noPlaceItems.length}, " +
      s"original place items: ${originalPlaceItems.length}")

    // projectionPi doesn't contain these leaf codes, so original places are kept
    originalPlaceItems.foreach { case (itemId, _) =>
      val oldCode = idCodeMap(itemId)
      projectionPi(itemId) = oldCode
      unAssignedLeafCodes -= oldCode
    }

    // assign to nearest leaf code
    noPlaceItems.foreach { case (itemId, code) =>
      val leafCode = code * 2 + 1
      val nearestCode = unAssignedLeafCodes.reduce { (a, b) =>
        if (math.abs(a - leafCode) < math.abs(b - leafCode)) a else b
      }
      projectionPi(itemId) = nearestCode
      unAssignedLeafCodes -= nearestCode
    }

    println("unassigned nodes remained: " + unAssignedLeafCodes.size)
    assert(unAssignedLeafCodes.isEmpty, "still remains unassigned codes")
  }
}
