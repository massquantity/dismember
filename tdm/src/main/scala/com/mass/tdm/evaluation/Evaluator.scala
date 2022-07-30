package com.mass.tdm.evaluation

import com.mass.scalann.{Criterion, Module}
import com.mass.scalann.tensor.Tensor
import com.mass.scalann.utils.{Engine, Table}
import com.mass.tdm.dataset.LocalDataSet
import com.mass.tdm.dataset.MiniBatch.{MaskTransformedBatch, SeqTransformedBatch}
import com.mass.tdm.evaluation.Metrics.computeMetrics
import com.mass.tdm.model.Recommender
import com.mass.tdm.operator.TDMOp

object Evaluator extends Serializable with Recommender {

  def evaluate(
      models: Array[Module[Float]],
      dataset: LocalDataSet,
      criterions: Array[Criterion[Float]],
      topk: Int,
      candidateNum: Int,
      state: Table,
      useMask: Boolean): EvalResult = {

    val subModelNum = Engine.coreNumber()
    val miniBatchIter = dataset.iteratorMiniBatch(train = false, expandBatch = true)

    miniBatchIter.map { batch =>
      val allData = dataset.getEvalData
      val userConsumed = dataset.getUserConsumed
      val miniBatchSize = batch.getLength
      val taskSize = miniBatchSize / subModelNum
      val extraSize = miniBatchSize % subModelNum
      val realParallelism = if (taskSize == 0) extraSize else subModelNum

      Engine.default.invokeAndWait(
        (0 until realParallelism).map(i => () => {
          val offset = batch.getOffset + i * taskSize + math.min(i, extraSize)
          val length = taskSize + (if (i < extraSize) 1 else 0)
          val transformedBatch = batch.convert(allData, offset, length, i)
          val (inputs, targets) = transformedBatch match {
            case m: SeqTransformedBatch =>
              (Table(m.items, m.sequence), m.labels)
            case m: MaskTransformedBatch =>
              (Table(m.items, m.sequence, m.masks), m.labels)
          }
          val localModel = models(i)
          localModel.evaluate()
          val outputs = localModel.forward(inputs).asInstanceOf[Tensor[Float]]
          val localLoss = criterions(i).forward(outputs, targets).toDouble
          val evalResult = new EvalResult(loss = localLoss * length, count = length)

          var j = offset
          while (j < offset + length) {
            val consumedItems = Some(userConsumed(allData(j).user))
            val labels = allData(j).labels
            val recItems = recommendItems(allData(j).sequence, localModel,
              TDMOp.tree, topk, candidateNum, useMask, consumedItems)
            evalResult += computeMetrics(recItems, labels)
            j += 1
          }
          evalResult
        })
      ).reduce(_ + _)  // reduce in one batch
    }.reduce(_ + _)   // reduce in all data
  }
}
