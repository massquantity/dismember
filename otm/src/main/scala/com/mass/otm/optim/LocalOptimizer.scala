package com.mass.otm.optim

import scala.collection.mutable

import com.mass.otm.{lowerLog2, upperLog2, DeepModel}
import com.mass.otm.dataset.{LocalDataSet, MiniBatch, OTMSample}
import com.mass.otm.evaluation.Evaluator
import com.mass.otm.model.ModelUtil._
import com.mass.otm.tree.OTMTree
import com.mass.otm.tree.OTMTree.{BatchNodes, Node}
import com.mass.scalann.nn.BCECriterionWithLogits
import com.mass.scalann.optim.Adam
import com.mass.scalann.tensor.Tensor
import com.mass.scalann.tensor.TensorNumeric.NumericDouble
import com.mass.scalann.utils.Engine
import org.apache.log4j.Logger

class LocalOptimizer(
    model: DeepModel[Double],
    dataset: LocalDataSet,
    targetMode: String,
    numEpoch: Int,
    totalTrainBatchSize: Int,
    totalEvalBatchSize: Int,
    learningRate: Double,
    beamSize: Int,
    topk: Int,
    seqLen: Int,
    useMask: Boolean,
    progressInterval: Int
) {
  import LocalOptimizer._
  val logger: Logger = Logger.getLogger(getClass)

  val criterion = BCECriterionWithLogits[Double]()
  val adamOptimizer = Adam[Double](learningRate)
  val (startLevel, leafLevel) = (lowerLog2(beamSize), upperLog2(dataset.numItem))
  val trainBatchSize = math.max(1, totalTrainBatchSize / (beamSize * 2))
  val numBatchInEpoch = math.ceil(dataset.trainSize.toDouble / trainBatchSize).toInt
  val tree = OTMTree(startLevel, leafLevel)
  val numThread = Engine.coreNumber()

  val (deepModel, clonedModels) = duplicateModels(model, numThread)
  val clonedGradients = clonedModels.map(getParameters(_)._2)
  val clonedCriterions = (1 to numThread).map(_ => criterion.cloneCriterion())

  logger.info(
    s"Overall train data size: ${dataset.trainSize}, " +
      s"true train batch size: $trainBatchSize, " +
      s"batch num in epoch: $numBatchInEpoch, " +
      s"tree start level: $startLevel, " +
      s"tree leaf level: $leafLevel"
  )

  def optimize(): Map[String, List[List[Double]]] = {
    (1 to numEpoch).foldLeft(Map.empty[String, List[List[Double]]]) { (epochResults, epoch) =>
      var epochTime, dataCount = 0L
      val trainData = scala.util.Random.shuffle(dataset.trainData)
      val epochLoss = trainData
        .sliding(trainBatchSize, trainBatchSize)
        .zip(1 to numBatchInEpoch)
        .map { case (batchData, iter) =>
          val start = System.nanoTime()
          implicit val (preModels, _seqLen, _useMask) = (clonedModels, seqLen, useMask)
          val threadData = splitToThreadData(batchData, numThread)
          val targetNodes = targetMode match {
            case "pseudo" => tree.optimalPseudoTargets(threadData)
            case "normal" => tree.normalTargets(threadData)
          }
          val beamSearchNodes = tree.beamSearchNodes(threadData, beamSize)
          val miniBatches = threadData.map(td => MiniBatch(td, beamSize, startLevel))
          val initInfo = LevelInfo(Nil, startLevel)
          val LevelInfo(batchLoss, _) = targetNodes.zip(beamSearchNodes).foldLeft(initInfo) {
            case (LevelInfo(levelLoss, level), (levelTargetNodes, levelBeamNodes)) =>
              val beamStart = if (level == startLevel) true else false
              val loss = trainBatch(levelTargetNodes, levelBeamNodes, miniBatches, beamStart)
              val (totalWeights, totalGradients) = syncGradients(threadData.length)
              adamOptimizer.optimize(_ => (loss, totalGradients), totalWeights)
              LevelInfo(loss :: levelLoss, level + 1)
          }
          val end = System.nanoTime()
          val iterationTime = end - start
          epochTime += iterationTime
          dataCount += batchData.length

          if (iter == numBatchInEpoch || (progressInterval > 0 && iter % progressInterval == 0)) {
            val iterMetrics = reportProgress(
              clonedModels,
              dataset,
              tree,
              topk,
              totalEvalBatchSize,
              beamSize,
              seqLen,
              useMask,
              epoch,
              iter,
              dataCount,
              epochTime,
              iterationTime,
              batchLoss.head
            )
            logger.info(iterMetrics)
          }
          batchLoss.reverse
        }
      epochResults + (s"epoch $epoch" -> epochLoss.toList.transpose)
    }
  }

  def trainBatch(
      targetNodes: IndexedSeq[BatchNodes],
      beamNodes: IndexedSeq[BatchNodes],
      miniBatches: IndexedSeq[MiniBatch],
      beamStart: Boolean
  ): Double = {
    val totalLoss = Engine.default.invokeAndWait(miniBatches.indices.map { i => () =>
      val transformedData = miniBatches(i).batchTransform(beamNodes(i), targetNodes(i), beamStart)
      val localModel = clonedModels(i)
      localModel.zeroGradParameters()
      localModel.training()
      val (inputs, labels) = transformedData
      val outputs = localModel.forward(inputs).toTensor
      val loss = clonedCriterions(i).forward(outputs, labels)
      val gradients = clonedCriterions(i).backward(outputs, labels)
      localModel.backward(inputs, gradients)
      loss
    })
    totalLoss.sum / miniBatches.length
  }

  def syncGradients(syncNum: Int): (Tensor[Double], Tensor[Double]) = {
    val (totalWeights, totalGradients) = getParameters(deepModel)
    (0 until syncNum).foreach {
      case i @ 0 => totalGradients.copy(clonedGradients(i))
      case i @ _ => totalGradients.add(clonedGradients(i))
    }
    totalGradients.div(syncNum)
    (totalWeights, totalGradients)
  }

  def searchCandidates(
      batchData: Seq[OTMSample],
      candidateNodes: Seq[Seq[Int]],
      candidateNum: Int,
      batchOutputs: Seq[Tensor[Double]]
  ): IndexedSeq[Seq[Node]] = {
    val threadDataSize = math.ceil(batchData.length.toDouble / numThread).toInt
    candidateNodes
      .sliding(threadDataSize, threadDataSize)
      .zip(batchOutputs)
      .map { case (threadNodes, threadOutputs) =>
        // take certain size since the underlying array may be larger than u think
        val offset = threadOutputs.storageOffset()
        val end = offset + threadOutputs.nElement()
        threadOutputs
          .storage()
          .array()
          .slice(offset, end)
          .sliding(candidateNum, candidateNum)
          .zip(threadNodes)
          .map { case (preds, nodes) =>
            nodes
              .lazyZip(preds)
              .map(Node)
              .sortBy(_.score)(Ordering[Double].reverse)
              .take(beamSize)
          }
          .toVector
      }
      .reduce(_ ++ _)
  }
}

object LocalOptimizer {

  case class LevelInfo(loss: List[Double], level: Int)

  def apply(
      deepModel: DeepModel[Double],
      dataset: LocalDataSet,
      targetMode: String,
      numEpoch: Int,
      totalTrainBatchSize: Int,
      totalEvalBatchSize: Int,
      learningRate: Double,
      beamSize: Int,
      topk: Int,
      seqLen: Int,
      useMask: Boolean,
      progressInterval: Int
  ): LocalOptimizer = {
    new LocalOptimizer(
      deepModel,
      dataset,
      targetMode,
      numEpoch,
      totalTrainBatchSize,
      totalEvalBatchSize,
      learningRate,
      beamSize,
      topk,
      seqLen,
      useMask,
      progressInterval
    )
  }

  private def splitToThreadData(
      data: List[OTMSample],
      numThread: Int
  ): IndexedSeq[List[OTMSample]] = {
    val threadDataSize = math.ceil(data.length.toDouble / numThread).toInt
    data.sliding(threadDataSize, threadDataSize).toIndexedSeq
  }

  private def duplicateModels(
      model: DeepModel[Double],
      numThread: Int
  ): (DeepModel[Double], IndexedSeq[DeepModel[Double]]) = {
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
    (model, clonedModels)
  }

  private def reportProgress(
      models: IndexedSeq[DeepModel[Double]],
      dataset: LocalDataSet,
      tree: OTMTree,
      topk: Int,
      totalEvalBatchSize: Int,
      beamSize: Int,
      seqLen: Int,
      useMask: Boolean,
      epoch: Int,
      iteration: Int,
      dataCount: Long,
      epochTime: Long,
      iterationTime: Long,
      trainLoss: Double
  ): String = {
    val progressInfo = new mutable.StringBuilder
    progressInfo ++= f"Epoch $epoch Train time: ${epochTime / 1e9d}%.4fs, "
    progressInfo ++= s"count/total: $dataCount/${dataset.trainSize}, "
    progressInfo ++= f"Iteration $iteration time: ${iterationTime / 1e9d}%.4fs, "
    progressInfo ++= f"Train loss: $trainLoss%.4f\n"

    val evalStart = System.nanoTime()
    val (evalLoss, evalMetrics) = Evaluator.evaluate(
      models,
      dataset,
      tree,
      topk,
      totalEvalBatchSize,
      beamSize,
      seqLen,
      useMask
    )
    val evalEnd = System.nanoTime()
    progressInfo ++= f"\tEval time: ${(evalEnd - evalStart) / 1e9d}%.4fs, "
    progressInfo ++= f"Eval Loss: $evalLoss%.4f, "
    progressInfo ++= f"Metrics: $evalMetrics\n"
    progressInfo.toString
  }
}
