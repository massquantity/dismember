package com.mass.tdm.evaluation

import com.mass.sparkdl.{Criterion, Module}
import com.mass.sparkdl.parameters.AllReduceParameter
import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.utils.{Engine, Table}
import com.mass.tdm.dataset.{DistDataSet, LocalDataSet, TDMSample}
import com.mass.tdm.evaluation.Metrics.computeMetrics
import com.mass.tdm.model.Recommender
import com.mass.tdm.operator.TDMOp
import com.mass.tdm.optim.OptimUtil.Cache
import org.apache.spark.rdd.RDD

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
          val (inputs, targets) = batch.convert(allData, offset, length, i)
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

  def evaluateRDD(
      models: RDD[Cache[Float]],
      dataset: DistDataSet,
      parameters: AllReduceParameter[Float],
      topk: Int,
      candidateNum: Int,
      state: Table,
      useMask: Boolean): EvalResult = {

    val subModelNum = Engine.coreNumber()
    val evalDataRDD = dataset.originalEvalRDD()
    val miniBatchRDD = dataset.iteratorMiniBatch(train = false, expandBatch = true)

    miniBatchRDD.zipPartitions(evalDataRDD, models) { (miniBatchIter, dataIter, modelIter) =>
      val cachedModel: Cache[Float] = modelIter.next()
      val data: Array[TDMSample] = dataIter.next()
      // put updated parameters from server to local model
      parameters.getWeights(cachedModel.modelWeights.head).waitResult()

      val evalLocal = miniBatchIter.map { batch =>
        val miniBatchSize = batch.getLength
        val taskSize = miniBatchSize / subModelNum
        val extraSize = miniBatchSize % subModelNum
        val realParallelism = if (taskSize == 0) extraSize else subModelNum

        Engine.default.invokeAndWait(
          (0 until realParallelism).map(i => () => {
            val offset = batch.getOffset + i * taskSize + math.min(i, extraSize)
            val length = taskSize + (if (i < extraSize) 1 else 0)
            val (inputs, targets) = batch.convert(data, offset, length, i)
            val localModel = cachedModel.localModels(i)
            localModel.evaluate()
            val outputs = localModel.forward(inputs).asInstanceOf[Tensor[Float]]
            val localCriterion = cachedModel.localCriterions(i)
            val localLoss = localCriterion.forward(outputs, targets).toDouble
            val evalResult = new EvalResult(loss = localLoss * length, count = length)

            var j = offset
            val end = offset + length
            while (j < end) {
              val recItems = recommendItems(data(j).sequence, localModel,
                TDMOp.tree, topk, candidateNum, useMask)
              val labels = data(j).labels
              evalResult += computeMetrics(recItems, labels)
              j += 1
            }
            evalResult
          })
        ).reduce(_ + _)  // reduce in one batch
      }.reduce(_ + _)   // reduce in one partition
      Iterator.single(evalLocal)
    }.reduce(_ + _)    // RDD reduce in all data
  }
}
