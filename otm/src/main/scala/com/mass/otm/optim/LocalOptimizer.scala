package com.mass.otm.optim

import com.mass.otm.{lowerLog2, upperLog2, DeepModel}
import com.mass.otm.dataset.{LocalDataSet, MiniBatch, OTMSample}
import com.mass.otm.evaluation.Evaluator
import com.mass.otm.model.ModelUtil._
import com.mass.otm.tree.OTMTree
import com.mass.otm.tree.OTMTree.Node
import com.mass.sparkdl.nn.BCECriterionWithLogits
import com.mass.sparkdl.optim.Adam
import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.tensor.TensorNumeric.NumericDouble
import com.mass.sparkdl.utils.{Engine, Table}
import org.apache.log4j.Logger

class LocalOptimizer(
    model: DeepModel[Double],
    dataset: LocalDataSet,
    numEpoch: Int,
    totalTrainBatchSize: Int,
    totalEvalBatchSize: Int,
    learningRate: Double,
    beamSize: Int,
    topk: Int,
    seqLen: Int,
    useMask: Boolean,
    progressInterval: Int) {
  import LocalOptimizer._
  val logger: Logger = Logger.getLogger(getClass)

  val criterion = BCECriterionWithLogits[Double]()
  val adamOptimizer = Adam[Double](learningRate)
  val startLevel = lowerLog2(beamSize)
  val leafLevel = upperLog2(dataset.numItem)
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

  // todo: report last level train loss only?
  def optimize(): Map[String, List[List[Double]]] = {
    (1 to numEpoch).foldLeft(Map.empty[String, List[List[Double]]]) { (epochResults, epoch) =>
      var epochTime, dataCount = 0L
      val trainData = scala.util.Random.shuffle(dataset.trainData)
      val epochLoss = trainData
        .sliding(trainBatchSize, trainBatchSize)
        .zip(1 to numBatchInEpoch)
        .map { case (batchData, iter) =>
          val start = System.nanoTime()
          implicit val preModel = deepModel  // .cloneModule()
          val pseudoTargets = tree.optimalPseudoTargets(batchData, seqLen, useMask)
          // val pseudoTargets = tree.normalTargets(batchData)
          val (initCandidates, initSize) = tree.initializeBeam(batchData.length)
          val miniBatch = MiniBatch(batchData, initSize, beamSize, seqLen, useMask, numThread)
          val initInfo = LevelInfo(initCandidates, Nil, startLevel)
          val LevelInfo(_, batchLoss, _) = pseudoTargets.foldLeft(initInfo) {
            case (LevelInfo(parentNodes, levelLoss, level), targets) =>
              val candidateNodes = parentNodes.map(_.flatMap(i => Seq(i.id * 2 + 1, i.id * 2 + 2)))
              val transformedData = levelLoss match {
                case Nil => miniBatch.batchTransform(candidateNodes, targets, initSize)
                case _ => miniBatch.batchTransform(candidateNodes, targets, beamSize)
              }
              val (batchOutputs, loss) = trainBatch(transformedData)
              val (totalWeights, totalGradients) = syncGradients()
              adamOptimizer.optimize(_ => (loss, totalGradients), totalWeights)

              val candidateNum = if (level == startLevel) initSize * 2 else beamSize * 2
              val levelNodes = searchCandidates(batchData, candidateNodes, candidateNum, batchOutputs)
              LevelInfo(levelNodes, loss :: levelLoss, level + 1)
          }
          val end = System.nanoTime()
          val iterationTime = end - start
          epochTime += iterationTime
          dataCount += batchData.length

          if (iter == numBatchInEpoch || iter % progressInterval == 0) {
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

  def trainBatch(transformedData: IndexedSeq[(Table, Tensor[Double])]): (Seq[Tensor[Double]], Double) = {
    val trainResults = Engine.default.invokeAndWait(
      (0 until numThread).map { i => () =>
        val localModel = clonedModels(i)
        localModel.zeroGradParameters()
        localModel.training()
        val (inputs, labels) = transformedData(i)
        val outputs = localModel.forward(inputs).toTensor
        val loss = clonedCriterions(i).forward(outputs, labels)
        val gradients = clonedCriterions(i).backward(outputs, labels)
        localModel.backward(inputs, gradients)
        (outputs, loss)
      }
    ).unzip
    (trainResults._1, trainResults._2.sum / numThread)
  }

  def syncGradients(): (Tensor[Double], Tensor[Double]) = {
    val (totalWeights, totalGradients) = getParameters(deepModel)
    (0 until numThread).foreach {
      case i@0 => totalGradients.copy(clonedGradients(i))
      case i@_ => totalGradients.add(clonedGradients(i))
    }
    totalGradients.div(numThread)
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
              .sortBy(_.pred)(Ordering[Double].reverse)
              .take(beamSize)
          }.toVector
      }.reduce(_ ++ _)
  }
}

object LocalOptimizer {

  case class LevelInfo(candidateNodes: Seq[Seq[Node]], loss: List[Double], level: Int)

  def apply(
    deepModel: DeepModel[Double],
    dataset: LocalDataSet,
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
    val progressInfo = new StringBuilder
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
