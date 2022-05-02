package com.mass.otm.optim

import com.mass.otm.{lowerLog2, upperLog2, DeepModel}
import com.mass.otm.dataset.{LocalDataSet, MiniBatch}
import com.mass.otm.evaluation.Evaluator
import com.mass.otm.tree.OTMTree
import com.mass.otm.tree.OTMTree.Node
import com.mass.sparkdl.nn.BCECriterionWithLogits
import com.mass.sparkdl.optim.Adam
import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.tensor.TensorNumeric.NumericDouble
import com.mass.sparkdl.utils.Table
import org.apache.log4j.Logger

class LocalOptimizer(
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
  val (totalWeights, totalGradients) = deepModel.adjustParameters()

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
          implicit val preModel = deepModel.cloneModule()
          val pseudoTargets = tree.optimalPseudoTargets(batchData, seqLen, useMask)
          val (itemSeqs, masks) = MiniBatch.duplicateSequence(batchData, beamSize, seqLen)
          val (initCandidates, initSize) = tree.initializeBeam(batchData.length)
          val LevelInfo(_, batchLoss) = pseudoTargets.foldLeft(LevelInfo(initCandidates, Nil)) {
            case (LevelInfo(parentNodes, levelLoss), targets) =>
              val candidateNodes = parentNodes.map(_.flatMap(i => Seq(i.id * 2 + 1, i.id * 2 + 2)))
              val transformFunc = MiniBatch.transform(batchData, candidateNodes, targets, seqLen, useMask) _
              val (batchInputs, batchLabels, candidateNum) = levelLoss match {
                case Nil => transformFunc(None, initSize)
                case _ => transformFunc(Some((itemSeqs, masks)), beamSize)
              }
              val (batchOutputs, loss) = trainBatch(batchInputs, batchLabels)
              val candidatePreds = batchOutputs
                .storage()
                .array()
                .sliding(candidateNum, candidateNum)
                .toSeq
              val levelNodes =
                for {
                  (nodes, preds) <- candidateNodes zip candidatePreds
                } yield {
                  nodes
                    .lazyZip(preds)
                    .map(Node)
                    .sortBy(_.pred)(Ordering[Double].reverse)
                    .take(beamSize)
                }
              LevelInfo(levelNodes, loss :: levelLoss)
          }
          val end = System.nanoTime()
          val iterationTime = end - start
          epochTime += iterationTime
          dataCount += batchData.length

          if (iter == numBatchInEpoch || iter % progressInterval == 0) {
            reportProgress(
              deepModel,
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
              batchLoss.head,
              logger
            )
          }
          batchLoss.reverse
        }
      epochResults + (s"epoch $epoch" -> epochLoss.toList.transpose)
    }
  }

  def trainBatch(inputs: Table, labels: Tensor[Double]): (Tensor[Double], Double) = {
    deepModel.zeroGradParameters()
    deepModel.training()
    val outputs = deepModel.forward(inputs).toTensor
    val loss = criterion.forward(outputs, labels)
    val gradients = criterion.backward(outputs, labels)
    deepModel.backward(inputs, gradients)
    adamOptimizer.optimize(_ => (loss, totalGradients), totalWeights)
    (outputs, loss)
  }
}

object LocalOptimizer {

  case class LevelInfo(candidateNodes: Seq[Seq[Node]], loss: List[Double])

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

  private def reportProgress(
    model: DeepModel[Double],
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
    trainLoss: Double,
    logger: Logger
  ): Unit = {
    val progressInfo = new StringBuilder
    progressInfo ++= f"Epoch $epoch Train time: ${epochTime / 1e9d}%.4fs, "
    progressInfo ++= s"count/total: $dataCount/${dataset.trainSize}, "
    progressInfo ++= f"Iteration $iteration time: ${iterationTime / 1e9d}%.4fs, "
    progressInfo ++= f"Train loss: $trainLoss%.4f\n"

    val evalStart = System.nanoTime()
    val (evalLoss, evalMetrics) = Evaluator.evaluate(
      model,
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
    logger.info(progressInfo.toString)
  }
}
