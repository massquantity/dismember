package com.mass.dr.optim

import com.mass.dr.dataset.{LocalDataSet, MiniBatch}
import com.mass.dr.dataset.MiniBatch.{LayerTransformedBatch, RerankTransformedBatch}
import com.mass.dr.model.DeepRetrieval
import com.mass.sparkdl.{Criterion, Module}
import com.mass.sparkdl.nn.{CrossEntropyCriterion, SampledSoftmaxLoss}
import com.mass.sparkdl.optim.{Adam, OptimMethod}
import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.tensor.TensorNumeric.NumericDouble
import com.mass.sparkdl.utils.{Engine, T, Util}
import org.apache.log4j.Logger

class LocalOptimizer(
    dataset: LocalDataSet,
    drModel: DeepRetrieval,
    numIteration: Int,
    numLayer: Int,
    learningRate: Double,
    numSampled: Int,
    embedSize: Int) {
  type LayerMulti[+T] = Seq[IndexedSeq[T]]
  val logger: Logger = Logger.getLogger(getClass)

  val (layerModel, reRankModel) = (drModel.layerModel, drModel.reRankModel)
  val (layerCriterion, layerOptimizer) = (1 to numLayer).map { _ =>
    (CrossEntropyCriterion(), Adam[Double](learningRate))
  }.unzip
  val reRankCriterion: SampledSoftmaxLoss[Double] = SampledSoftmaxLoss[Double](
    numSampled,
    dataset.numItem,
    embedSize,
    learningRate,
    drModel.reRankWeights,
    drModel.reRankBias
  )
  val reRankOptimizer: OptimMethod[Double] = Adam[Double](learningRate)

  private val subModelNum = Engine.coreNumber()
  private var realParallelism: Int = _

  lazy val layerCopiedModels: LayerMulti[Module[Double]] = {
    // make the parameters compact
    layerModel.foreach(_.adjustParameters())
    val weights: LayerMulti[Tensor[Double]] = layerModel.map { model =>
      Util.getAndClearWeightBias(model.parameters()).toIndexedSeq
    }

    // all models share same weight
    val models: LayerMulti[Module[Double]] = layerModel.zip(weights) map { case (model, w) =>
      (1 to subModelNum).map { _ =>
        val m = model.cloneModule()
        Util.putWeightBias(w.toArray, m)
        Util.initGradWeightBias(w.toArray, m)
        m
      }
    }

    layerModel.zip(weights).foreach { case (model, w) =>
      Util.putWeightBias(w.toArray, model)
      Util.initGradWeightBias(w.toArray, model)
    }
    models
  }

  lazy val (totalLayerWeights, totalLayerGradients) = {
    val parameters = layerModel.map(_.adjustParameters())
    (parameters.map(_._1), parameters.map(_._2))
  }
  lazy val layerCopiedGradients: LayerMulti[Tensor[Double]] = {
    // layerCopiedModels.map(_.map(_.adjustParameters()._2))
    val grads = for {
      models <- layerCopiedModels
      m <- models
      grad = m.adjustParameters()._2
    } yield grad
    grads.toIndexedSeq.sliding(subModelNum, subModelNum).toSeq
  }
  lazy val layerCopiedCriterions: LayerMulti[Criterion[Double]] = {
    // layerCriterion.map(c => (1 to subModelNum).map(_ => c.cloneCriterion()))
    val criterions = for {
      criterion <- layerCriterion
      _ <- 1 to subModelNum
    } yield criterion.cloneCriterion()
    criterions.toIndexedSeq.sliding(subModelNum, subModelNum).toSeq
  }

  def optimize(): Unit = {
    layerOptimizer.foreach(_.clearHistory())
    reRankOptimizer.clearHistory()

    dataset.shuffle()
    val miniBatchIter = dataset.iteratorMiniBatch(train = true)
    (0 until numIteration) foreach { iter =>
      val batch: MiniBatch = miniBatchIter.next()
      // train layer model
      val layerBatch = transformLayerBatch(batch)
      val layerLoss = trainLayerBatch(layerBatch)
      syncGradients()
      (0 until numLayer) foreach { i =>
        layerOptimizer(i).optimize(_ => (layerLoss(i), totalLayerGradients(i)), totalLayerWeights(i))
      }

      // train rerank model
      val reRankLoss = trainRerank(batch)
      val (reRankWeights, reRankGradients) = reRankModel.adjustParameters()
      reRankOptimizer.optimize(_ => (reRankLoss, reRankGradients), reRankWeights)
      logger.info(s"Iteration $iter, layer loss: $layerLoss, rerank loss: $reRankLoss")
    }
    shutdown()
  }

  private def trainRerank(batch: MiniBatch): Double = {
    val reRankBatch: RerankTransformedBatch = batch.transformRerankData(
      dataset.getTrainData, batch.getOffset, batch.getLength
    )
    val inputs = reRankBatch.itemSeqs
    val labels = reRankBatch.target.toTensor[Double]  // asInstanceOf[Tensor[Double]]
    val outputs = reRankModel.forward(inputs).toTensor
    val reRankLoss = reRankCriterion.forward(outputs, labels)
    val lossGrad = reRankCriterion.backward(outputs, labels)
    reRankModel.backward(inputs, lossGrad)
    reRankLoss
  }

  private def transformLayerBatch(batch: MiniBatch): Seq[LayerTransformedBatch] = {
    val allData = dataset.getTrainData
    val miniBatchOffset = batch.getOffset
    val miniBatchSize = batch.getLength
    val taskSize = miniBatchSize / subModelNum
    val extraSize = miniBatchSize % subModelNum
    realParallelism = if (taskSize == 0) extraSize else subModelNum
    Engine.default.invokeAndWait(
      (0 until realParallelism).map(i => () => {
        val offset = miniBatchOffset + i * taskSize + math.min(i, extraSize)
        val length = taskSize + (if (i < extraSize) 1 else 0)
        batch.transformLayerData(allData, offset, length)
      })
    )
  }

  private def trainLayerBatch(miniBatch: Seq[LayerTransformedBatch]): Seq[Double] = {
    layerCopiedModels
      .zip(layerCopiedCriterions)
      .zipWithIndex
      .map { case ((models, criterions), layerIdx) =>
        val lossSum = Engine.default.invokeAndWait(
          (0 until realParallelism).map(i => () => {
            val localModel = models(i)
            localModel.zeroGradParameters()
            localModel.training()
            val localCriterion = criterions(i)
            val inputs =
              if (layerIdx == 0) {
                miniBatch(i).itemSeqs
              } else {
                T(miniBatch(i).itemSeqs, miniBatch(i).paths(layerIdx - 1))
              }
            val labels = miniBatch(i).targets(layerIdx).toTensor
            val outputs = localModel.forward(inputs).toTensor
            val localLoss = localCriterion.forward(outputs, labels)
            val gradients = localCriterion.backward(outputs, labels)
            localModel.backward(inputs, gradients)
            localLoss
        })
      ).sum
      lossSum / realParallelism
    }
  }

  private def syncGradients(): Unit = {
    totalLayerGradients.zipWithIndex.foreach { case (grad, gradIdx) =>
      val gradLength = grad.nElement()
      val syncGradTaskSize = gradLength / subModelNum
      val syncGradExtraSize = gradLength % subModelNum
      val syncGradParallelNum = if (syncGradTaskSize == 0) syncGradExtraSize else subModelNum
      Engine.default.invokeAndWait(
        (0 until syncGradParallelNum).map(tid => () => {
          val offset = tid * syncGradTaskSize + math.min(tid, syncGradExtraSize)
          val length = syncGradTaskSize + (if (tid < syncGradExtraSize) 1 else 0)
          (0 until realParallelism) foreach {
            case i@0 =>
              grad.narrow(0, offset, length)
                .copy(layerCopiedGradients(gradIdx)(i).narrow(0, offset, length))
            case i@_ =>
              grad.narrow(0, offset, length)
                .add(layerCopiedGradients(gradIdx)(i).narrow(0, offset, length))
          }
        })
      )
      grad.div(realParallelism)
    }
  }

  def shutdown(): Unit = {
    layerCopiedModels.foreach(_.foreach(_.release()))
  }

}
