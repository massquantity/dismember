package com.mass.tdm.model

import scala.annotation.tailrec
import scala.collection.mutable.ArrayBuffer
import scala.collection.View
import scala.math.Ordering

import com.mass.sparkdl.Module
import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.tensor.TensorNumeric.NumericFloat
import com.mass.sparkdl.utils.Table
import com.mass.tdm.protobuf.tree.Node
import com.mass.tdm.tree.TDMTree
import com.mass.tdm.tree.DistTree.TreeNode

trait Recommender {
  import Recommender._

  def recommendItems(
    sequence: Array[Int],
    model: Module[Float],
    tree: TDMTree,
    topk: Int,
    candidateNum: Int,
    useMask: Boolean,
    consumedItems: Option[Seq[Int]] = None
  ): Array[Int] = {
    // If a user has consumed many items, the candidate number may be large.
    val (_candidateNum, _consumedItems) = consumedItems match {
      case Some(items) =>
        (math.max((items.length + topk) / 2, candidateNum), items.toSet)
      case None =>
        (candidateNum, Set.empty[Int])
    }

    val recs = _recommend(sequence, model, tree, _candidateNum, useMask, _consumedItems)
    // recs.sorted(Ordering.by[TreeNodePred, Float](_.pred)(Ordering[Float].reverse))
    recs.sortBy(_._2)(Ordering[Float].reverse).take(topk).map(_._1)
  }

  private[model] def _recommend(
    sequence: Array[Int],
    model: Module[Float],
    tree: TDMTree,
    candidateNum: Int,
    useMask: Boolean,
    consumedItems: Set[Int],
  ): Array[(Int, Float)] = {
    val codeNodeMap = tree.codeNodeMap
    // binary tree with 2 child => candidateNum * 2
    val modelInputs = duplicateSequence(sequence, tree, useMask, candidateNum * 2)
    val (levelStartCode, level) = getLevelStart(candidateNum, 0, 0)
    val levelEndCode = levelStartCode * 2 + 1
    val initCandidates = Vector.range(levelStartCode, levelEndCode)
      .filter(codeNodeMap.contains)
      .map(i => TreeNodePred(i, codeNodeMap(i), 0.0f))
    val initValue = LevelInfo(initCandidates, Nil)
    val finalLevelInfo = (level until tree.maxLevel).foldLeft(initValue) { (levelInfo, _) =>
      if (levelInfo.candidateNodes.isEmpty) {
        levelInfo
      } else {
        val (leafNodes, nonLeafNodes) = levelInfo.candidateNodes.partition(n => codeNodeMap(n.code).isLeaf)
        val newLeafNodes =
          if (leafNodes.nonEmpty) {
            leafNodes ++: levelInfo.leafNodes
          } else {
            levelInfo.leafNodes
          }
        if (nonLeafNodes.isEmpty) {
          LevelInfo(Vector.empty[TreeNodePred], newLeafNodes)
        } else {
          val beamNodes =
            if (nonLeafNodes.length > candidateNum) {
              nonLeafNodes.sorted(
                new Ordering[TreeNodePred] {
                  override def compare(x: TreeNodePred, y: TreeNodePred): Int = {
                    y.pred.compareTo(x.pred)
                  }
                }
              ).take(candidateNum)
            } else {
              nonLeafNodes
            }
          val childrenNodes = beamNodes
            .view
            .flatMap(n => View(2 * n.code + 1, 2 * n.code + 2))
            .filter(codeNodeMap.contains)
            .map(i => TreeNode(i, codeNodeMap(i)))
            .toArray
          val features = modelInputs.buildInputs(childrenNodes)
          val preds = model.forward(features).toTensor.storage().array()
          val newCandidateNodes = Vector.range(0, childrenNodes.length).map { i =>
            TreeNodePred(childrenNodes(i).code, childrenNodes(i).node, preds(i))
          }
          LevelInfo(newCandidateNodes, newLeafNodes)
        }
      }
    }

    finalLevelInfo
      .leafNodes
      .view
      .filterNot(i => consumedItems.contains(i.node.id))
      .map(i => Tuple2(i.node.id, i.pred))
      .toArray
  }
}

object Recommender {

  case class TreeNodePred(code: Int, node: Node, pred: Float)

  case class LevelInfo(candidateNodes: Vector[TreeNodePred], leafNodes: List[TreeNodePred])

  sealed trait ModelInputs {

    def buildInputs(targetNodes: Array[TreeNode]): Table

    def narrowInputRange(
      targetNodes: Array[TreeNode],
      targetItem: Tensor[Int],
      seqItems: Tensor[Int]
    ): (Tensor[Int], Tensor[Int]) = {
      val num = targetNodes.length
      0 until num foreach { i =>
        targetItem.setValue(i, 0, targetNodes(i).code)
      }
      (targetItem.narrow(0, 0, num), seqItems.narrow(0, 0, num))
    }
  }

  private class SeqModelInputs(
      targetItem: Tensor[Int],
      seqItems: Tensor[Int]) extends ModelInputs {

    override def buildInputs(targetNodes: Array[TreeNode]): Table = {
      val (item, seq) = narrowInputRange(targetNodes, targetItem, seqItems)
      Table(item, seq)
    }
  }

  private class MaskModelInputs(
      targetItem: Tensor[Int],
      seqItems: Tensor[Int],
      masks: Tensor[Int],
      maskLen: Int) extends ModelInputs {

    override def buildInputs(targetNodes: Array[TreeNode]): Table = {
      val num = targetNodes.length
      val (item, seq) = narrowInputRange(targetNodes, targetItem, seqItems)
      // masks is one-dimensional tensor
      val mask = if (masks.isEmpty) masks else masks.narrow(0, 0, num * maskLen)
      Table(item, seq, mask)
    }
  }

  private def duplicateSequence(
      sequence: Array[Int],
      tree: TDMTree,
      useMask: Boolean,
      candidateLen: Int): ModelInputs = {
    val seqLen = sequence.length
    // dummy targets for later setting value
    val targets = new Array[Int](candidateLen)
    val targetItems = Tensor(targets, Array(candidateLen, 1))
    val features = new Array[Int](candidateLen * seqLen)
    val seqItems = Tensor(features, Array(candidateLen, seqLen))
    if (!useMask) {
      val (seqCodes, _) = tree.idToCode(sequence)
      val _seq = seqCodes.toArray
      var i = 0
      while (i < candidateLen) {
        val offset = i * seqLen
        System.arraycopy(_seq, 0, features, offset, seqLen)
        i += 1
      }
      new SeqModelInputs(targetItems, seqItems)
    } else {
      val (seqCodes, mask) = tree.idToCode(sequence)
      val _seq = seqCodes.toArray
      val maskBuffer = new ArrayBuffer[Int]()
      var i = 0
      while (i < candidateLen) {
        val offset = i * seqLen
        System.arraycopy(_seq, 0, features, offset, seqLen)
        mask.foreach(m => maskBuffer += (m + offset))
        i += 1
      }
      val masks =
        if (maskBuffer.isEmpty) {
          Tensor[Int]()
        } else {
          Tensor(maskBuffer.toArray, Array(maskBuffer.length))
        }
      new MaskModelInputs(targetItems, seqItems, masks, mask.length)
    }
  }

  @tailrec
  def getLevelStart(candidateNum: Int, n: Int, level: Int): (Int, Int) = {
    if (candidateNum == 0) (n, level)
    else getLevelStart((candidateNum - 1) / 2, n * 2 + 1, level + 1)
  }
}
