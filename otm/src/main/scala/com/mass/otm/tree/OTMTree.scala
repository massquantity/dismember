package com.mass.otm.tree

import scala.collection.mutable.ArrayBuffer

import com.mass.otm.{clipValue, paddingIdx, DeepModel}
import com.mass.otm.dataset.OTMSample
import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.tensor.TensorNumeric.NumericDouble
import com.mass.sparkdl.utils.{Engine, Table}

// start level has num of nodes <= beamSize, i.e. upper level of first candidate level
class OTMTree(val startLevel: Int, val leafLevel: Int) {
  import OTMTree._

  // batchSize * initSize
  def initializeBeam(batchSize: Int): (BatchNodes, Int) = {
    val startNode = (1 to startLevel).foldLeft(0)((i, _) => i * 2 + 1)
    val endNode = startNode * 2 + 1
    val initNodes = List.range(startNode, endNode).map(Node(_, 0.0))
    val initCandidates = List.fill(batchSize)(initNodes)
    val initSize = endNode - startNode
    (initCandidates, initSize)
  }

  // levelSize * numThread * (batchSize / numThread) * labelNum
  // line 5 in Algorithm 1 from the paper, bottom up computing pseudo targets
  def optimalPseudoTargets(threadData: IndexedSeq[List[OTMSample]])(
    implicit models: IndexedSeq[DeepModel[Double]],
    seqLen: Int,
    useMask: Boolean
  ): Seq[IndexedSeq[BatchNodes]] = {
    Engine.default.invokeAndWait(threadData.indices.map { i => () =>
      implicit val model = models(i)
      val batchData = threadData(i)
      (leafLevel until startLevel by -1).foldLeft[List[BatchNodes]](Nil) { (allNodes, _) =>
        val levelNodes = allNodes match {
          case Nil => batchData.map(_.targetItems.map(Node(_, 1.0)))
          case childNodes :: _ => computeTargets(childNodes, batchData)
        }
        levelNodes :: allNodes
      }
    }).toIndexedSeq.transpose
  }

  // levelSize * numThread * (batchSize / numThread) * labelNum
  // compute targets according to Eq.(1) from the paper
  def normalTargets(threadData: IndexedSeq[List[OTMSample]]): Seq[IndexedSeq[BatchNodes]] = {
    (
      for {
        batchData <- threadData
      } yield {
        batchData.map { d =>
          List.range(1, leafLevel - startLevel)
            .scanRight(d.targetItems)((_, items) => items.map(i => (i - 1) >> 1))
            .map(i => i.map(Node(_, 1.0)))
        }.transpose
      }
    ).transpose
  }

  // levelSize * numThread * (batchSize / numThread) * (beamSize * 2)
  // line 4 in Algorithm 1 from the paper, draw beam search nodes with fixed model parameter
  def beamSearchNodes(threadData: IndexedSeq[List[OTMSample]], beamSize: Int)(
    implicit models: IndexedSeq[DeepModel[Double]],
    seqLen: Int,
    useMask: Boolean
  ): Seq[IndexedSeq[BatchNodes]] = {
    Engine.default.invokeAndWait(threadData.indices.map { i => () =>
      implicit val model = models(i)
      val batchData = threadData(i)
      val (initCandidates, initSize) = initializeBeam(batchData.length)
      (startLevel until leafLevel).foldLeft[List[BatchNodes]](Nil) { case (allNodes, _) =>
        val levelNodes = allNodes match {
          case Nil =>
            computeBeamNodes(initCandidates, batchData, initSize, beamStart = true)
          case candidateNodes :: _ =>
            computeBeamNodes(candidateNodes, batchData, beamSize, beamStart = false)
        }
        levelNodes :: allNodes
      }.reverse
    }).toIndexedSeq.transpose
  }
}

object OTMTree {

  case class Node(id: Int, score: Double)

  type BatchNodes = List[List[Node]]

  def apply(startLevel: Int, leafLevel: Int): OTMTree = {
    new OTMTree(startLevel, leafLevel)
  }

  def computeTargets(
    childrenNodes: BatchNodes,  // batchSize * labelNum
    data: Seq[OTMSample]
  )(
    implicit model: DeepModel[Double],
    seqLen: Int,
    useMask: Boolean
  ): BatchNodes = {
    val labelNums = childrenNodes.map(_.length)
    val (itemSeqs, masks) = sequenceBatchLabels(data, labelNums)
    val (posPreds, negPreds, negLabels) = computeChildrenScores(childrenNodes, itemSeqs, masks)
    childrenNodes.foldRight[(List[List[Node]], Int)]((Nil, 0)) { case (nodes, (targets, offset)) =>
      val nodesWithLabels = nodes.zipWithIndex.map { case (n, i) =>
        val index = offset + i
        val label = if (posPreds(index) >= negPreds(index)) n.score else negLabels(index)
        (n.id, label)
      }
      val parentNodes = nodesWithLabels
        .groupMapReduce(i => (i._1 - 1) >> 1)(_._2)(_ + _)
        .toList
        .map(i => Node(i._1, clipValue(i._2, 0.0, 1.0)))
      (parentNodes :: targets, offset + nodes.length)
    }._1
  }

  def computeChildrenScores(
    batchNodes: List[List[Node]],
    itemSeqs: Tensor[Int],
    masks: Tensor[Int]
  )(
    implicit model: DeepModel[Double],
    useMask: Boolean
  ): (Array[Double], Array[Double], Array[Double]) = {
    val length = batchNodes.map(_.length).sum
    val batchPosNodes = ArrayBuffer[Int]()
    val batchNegNodes = ArrayBuffer[Int]()
    val negLabels = batchNodes.toArray.flatMap { nodes =>
      val posNodes = nodes.map(_.id)
      val negNodes = posNodes.map(n => if (n % 2 == 0) n - 1 else n + 1)
      batchPosNodes ++= posNodes
      batchNegNodes ++= negNodes
      negNodes.map { nn =>
        nodes.find(_.id == nn) match {
          case Some(i) => i.score
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
    val posPreds = computePreds(model, posInputs)
    val negPreds = computePreds(model, negInputs)
    (posPreds, negPreds, negLabels)
  }

  def computePreds(model: DeepModel[Double], input: Table): Array[Double] = {
    val predTensor = model.forward(input).toTensor
    val offset = predTensor.storageOffset()
    val end = offset + predTensor.nElement()
    predTensor.storage().array().slice(offset, end)
  }

  def computeBeamNodes(
    candidateNodes: BatchNodes,
    data: List[OTMSample],
    nodeSize: Int,
    beamStart: Boolean
  )(
    implicit model: DeepModel[Double],
    seqLen: Int,
    useMask: Boolean
  ): BatchNodes = {
    val beamNodes =
      if (beamStart) {
        candidateNodes.map(_.flatMap(n => List(n.id * 2 + 1, n.id * 2 + 2)))
      } else {
        for {
          nodes <- candidateNodes
        } yield {
          nodes
            .sortBy(_.score)(Ordering[Double].reverse)
            .take(nodeSize)
            .flatMap(n => List(n.id * 2 + 1, n.id * 2 + 2))
        }
      }
    val batchInputs = transformBeamData(beamNodes, data, nodeSize)
    val batchOutputs = model.forward(batchInputs).toTensor
    // take certain size since the underlying array may be larger than u think
    val offset = batchOutputs.storageOffset()
    val end = offset + batchOutputs.nElement()
    val candidatePreds = batchOutputs
      .storage()
      .array()
      .slice(offset, end)
      .sliding(nodeSize * 2, nodeSize * 2)
      .toSeq

    beamNodes.zip(candidatePreds).map { case (nodes, preds) =>
      nodes.lazyZip(preds).map(Node)
    }
  }

  def sequenceBatchLabels(
    data: Seq[OTMSample],
    labelNums: Seq[Int]
  )(implicit seqLen: Int): (Tensor[Int], Tensor[Int]) = {
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

  def transformBeamData(
    nodes: Seq[Seq[Int]],
    data: Seq[OTMSample],
    nodeSize: Int
  )(
    implicit seqLen: Int,
    useMask: Boolean
  ): Table = {
    val itemShape = Array(nodes.map(_.length).sum, 1)
    val itemTensor = Tensor(nodes.flatten.toArray, itemShape)
    val itemSeqShape = Array(data.length * nodeSize * 2, seqLen)
    val itemSeqs =
      for {
        d <- data.toArray
        _ <- 1 to (nodeSize * 2)
        i <- d.sequence
      } yield i
    val itemSeqTensor = Tensor(itemSeqs, itemSeqShape)
    val masks = itemSeqs.zipWithIndex.filter(_._1 == paddingIdx).map(_._2)
    val maskTensor = if (masks.isEmpty) Tensor[Int]() else Tensor(masks, Array(masks.length))
    if (useMask) {
      Table(itemTensor, itemSeqTensor, maskTensor)
    } else {
      Table(itemTensor, itemSeqTensor)
    }
  }
}
