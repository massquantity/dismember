package com.mass.otm.tree

import scala.collection.mutable.ArrayBuffer

import com.mass.otm.{paddingIdx, DeepModel}
import com.mass.otm.dataset.OTMSample
import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.tensor.TensorNumeric.NumericDouble
import com.mass.sparkdl.utils.{Engine, Table}

// start level has num of nodes <= beamSize, i.e. upper level of first candidate level
class OTMTree(val startLevel: Int, val leafLevel: Int) {
  import OTMTree._

  // batchSize * initSize
  def initializeBeam(batchSize: Int): (Seq[Seq[Node]], Int) = {
    val startNode = (1 to startLevel).foldLeft(0)((i, _) => i * 2 + 1)
    val endNode = startNode * 2 + 1
    val initNodes = Seq.range(startNode, endNode).map(Node(_, 0.0))
    val initCandidates = Seq.fill(batchSize)(initNodes)
    val initSize = endNode - startNode
    (initCandidates, initSize)
  }

  // levelSize * batchSize * labelNum
  def optimalPseudoTargets(data: List[OTMSample], seqLen: Int, useMask: Boolean, threadNum: Int)(
    implicit models: IndexedSeq[DeepModel[Double]]
  ): List[List[List[TargetNode]]] = {
    (leafLevel until startLevel by -1).foldLeft[List[List[List[TargetNode]]]](Nil) { (allNodes, _) =>
      val levelNodes = allNodes match {
        case Nil => data.map(_.targetItems.map(TargetNode(_, 1.0)))
        case childNodes :: _ => computeTargetsParallel(childNodes, data, seqLen, useMask, threadNum)
      }
      levelNodes :: allNodes
    }
  }

  // levelSize * batchSize * labelNum
  def normalTargets(data: Seq[OTMSample]): Seq[Seq[Seq[TargetNode]]] = {
    data.map { d =>
      Seq.range(1, leafLevel - startLevel)
        .scanRight(d.targetItems)((_, items) => items.map(i => (i - 1) >> 1))
        .map(i => i.map(TargetNode(_, 1.0)))
    }.transpose
  }
}

object OTMTree {

  case class Node(id: Int, pred: Double)

  case class TargetNode(id: Int, label: Double)

  val clipValue = (value: Double, min: Double, max: Double) => math.max(min, math.min(max, value))

  def apply(startLevel: Int, leafLevel: Int): OTMTree = {
    new OTMTree(startLevel, leafLevel)
  }

  def computeTargetsParallel(
    childrenNodes: List[List[TargetNode]],  // batchSize * labelNum
    batchData: Seq[OTMSample],
    seqLen: Int,
    useMask: Boolean,
    threadNum: Int
  )(
    implicit models: IndexedSeq[DeepModel[Double]]
  ): List[List[TargetNode]] = {
    val threadSize = math.ceil(batchData.length.toDouble / threadNum).toInt
    val threadNodes = childrenNodes.sliding(threadSize, threadSize).toSeq
    val threadData = batchData.sliding(threadSize, threadSize).toIndexedSeq
    Engine.default.invokeAndWait(
      threadNodes.zipWithIndex.map { case (nodes, i) => () =>
        implicit val model = models(i)
        val data = threadData(i)
        computeTargets(nodes, data, seqLen, useMask)
      }
    ).reduce(_ ::: _)
  }

  def computeTargets(
    childrenNodes: List[List[TargetNode]],
    data: Seq[OTMSample],
    seqLen: Int,
    useMask: Boolean
  )(
    implicit model: DeepModel[Double]
  ): List[List[TargetNode]] = {
    val labelNums = childrenNodes.map(_.length)
    val (itemSeqs, masks) = sequenceBatch(data, labelNums, seqLen)
    val (posPreds, negPreds, negLabels) = computeChildrenScores(childrenNodes, itemSeqs, masks, useMask)
    childrenNodes.foldRight[(List[List[TargetNode]], Int)]((Nil, 0)) { case (nodes, (targets, offset)) =>
      val labelNum = nodes.length
      val nodesWithLabels = nodes.zipWithIndex.map { case (n, i) =>
        val index = offset + i
        val label = if (posPreds(index) > negPreds(index)) n.label else negLabels(index)
        (n.id, label)
      }
      val parentNodes = nodesWithLabels
        .groupMapReduce(i => (i._1 - 1) >> 1)(_._2)(_ + _)
        .toList
        .map(i => TargetNode(i._1, clipValue(i._2, 0.0, 1.0)))
      (parentNodes :: targets, offset + labelNum)
    }._1
  }

  def computeChildrenScores(
    batchNodes: List[List[TargetNode]],
    itemSeqs: Tensor[Int],
    masks: Tensor[Int],
    useMask: Boolean
  )(
    implicit model: DeepModel[Double]
  ): (Array[Double], Array[Double], Array[Double]) = {
    val length = batchNodes.map(_.length).sum
    val batchPosNodes = ArrayBuffer[Int]()
    val batchNegNodes = ArrayBuffer[Int]()
    val negLabels = batchNodes.toArray.flatMap { nodes =>
      val posNodes = nodes.map(_.id)
      val negNodes = posNodes.map(n => if (n % 2 == 0) n - 1 else n + 1)
      batchPosNodes ++= posNodes
      batchNegNodes ++= negNodes
      negNodes.map { n =>
        nodes.find(_.id == n) match {
          case Some(i) => i.label
          case None => 0.0
        }
      }
    }
    val posTensor = Tensor(batchPosNodes.toArray, Array(length, 1))
    val negTensor = Tensor(batchNegNodes.toArray, Array(length, 1))
    val (posInputs, negInputs) =
      if (useMask) {
        (Table(posTensor, itemSeqs, masks), Table(negTensor, itemSeqs, masks))
      } else {
        (Table(negTensor, itemSeqs), Table(negTensor, itemSeqs))
      }
    val posPreds = model.forward(posInputs).toTensor.storage().array()
    val negPreds = model.forward(negInputs).toTensor.storage().array()
    (posPreds, negPreds, negLabels)
  }

  def sequenceBatch(
    data: Seq[OTMSample],
    labelNums: Seq[Int],
    seqLen: Int
  ): (Tensor[Int], Tensor[Int]) = {
    val sequence =
      for {
        (d, n) <- data.toArray zip labelNums
        _ <- 1 to n
        i <- d.sequence
      } yield i
    val masks = sequence.zipWithIndex.filter(_._1 == paddingIdx).map(_._2)
    val maskTensor = if (masks.isEmpty) Tensor[Int]() else Tensor(masks, Array(masks.length))
    (Tensor(sequence, Array(labelNums.sum, seqLen)), maskTensor)
  }
}
