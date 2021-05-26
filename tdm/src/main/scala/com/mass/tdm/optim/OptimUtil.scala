package com.mass.tdm.optim

import java.io.{File, FilenameFilter}

import scala.collection.mutable.ArrayBuffer
import scala.concurrent.Future

import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.utils.{Engine, Table}
import com.mass.sparkdl.{Criterion, Module}
import com.mass.sparkdl.nn.abstractnn.Activity
import com.mass.sparkdl.optim.OptimMethod
import com.mass.sparkdl.parameters.AllReduceParameter
import com.mass.tdm.dataset.{MiniBatch, TDMSample}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.util.{DoubleAccumulator, LongAccumulator}

object OptimUtil {

  case class Cache[T](
    localModels: Array[Module[T]],
    modelWeights: Array[Tensor[T]],
    modelGradients: Array[Tensor[T]],
    localCriterions: Array[Criterion[T]],
    var optimMethod: OptimMethod[T]
  )

  private[optim] def convertBatch(
      data: Array[TDMSample],
      batch: MiniBatch,
      parallel: Boolean,
      subModelNum: Int,
      tasks: ArrayBuffer[Future[_]])
    : (Array[(Activity, Tensor[Float])], Array[Int], Int) = {

    val miniBatchSize = if (parallel) batch.getLength else batch.expandedSize()
    val taskSize = miniBatchSize / subModelNum
    val extraSize = miniBatchSize % subModelNum
    val parallelism =  if (taskSize == 0) extraSize else subModelNum
    // miniBatch element in one thread: Tuple(feature, label)
    val miniBatchBuffer = new Array[(Activity, Tensor[Float])](parallelism)
    val miniBatchLen = new Array[Int](parallelism)

    if (parallel) {
      tasks ++= Engine.default.invoke {
        (0 until parallelism).map(i => () => {
          val offset = batch.getOffset + i * taskSize + math.min(i, extraSize)
          val length = taskSize + (if (i < extraSize) 1 else 0)
          miniBatchBuffer(i) = batch.convert(data, offset, length, i)
          miniBatchLen(i) = length
        })
      }
    } else {
      tasks += Engine.default.invoke( () => {
        batch.convertAll(data)
        var i = 0
        while (i < parallelism) {
          val offset = i * taskSize + math.min(i, extraSize)
          val length = taskSize + (if (i < extraSize) 1 else 0)
          miniBatchBuffer(i) = batch.slice(offset, length)
          miniBatchLen(i) = length
          i += 1
        }
      })
    }

    (miniBatchBuffer, miniBatchLen, parallelism)
  }

  private[optim]  def trainBatch(
      miniBatchBuffer: Array[(Activity, Tensor[Float])],
      miniBatchLen: Array[Int],
      cachedModel: Cache[Float],
      lossSum: DoubleAccumulator,
      recordsNum: LongAccumulator,
      parallelism: Int): Array[Int] = {

    val lossArray = new Array[Double](parallelism)
    val trainingThreads = Engine.default.invokeAndWait2(
      (0 until parallelism).map(i => () => {
        val localModel: Module[Float] = cachedModel.localModels(i)
        localModel.training()
        val localCriterion = cachedModel.localCriterions(i)
        val inputs = miniBatchBuffer(i)._1
        val labels = miniBatchBuffer(i)._2
        val output = localModel.forward(inputs).asInstanceOf[Tensor[Float]]
        lossArray(i) = localCriterion.forward(output, labels).toDouble
        val lastGrads = localCriterion.backward(output, labels)
        localModel.backward(inputs, lastGrads)
        i
      }))

    val finishedThreads = trainingThreads.filterNot(_.isCancelled).map(_.get())
    lossSum.add(finishedThreads.map(lossArray(_)).sum)
    recordsNum.add(finishedThreads.map(miniBatchLen(_)).sum.toLong)
    finishedThreads.toArray
  }

  private[optim] def syncGradients(
      finishedThreads: Array[Int],
      cachedModel: Cache[Float],
      parameters: AllReduceParameter[Float],
      subModelNum: Int): Unit = {

    if (finishedThreads.nonEmpty) {
      val finishedGradients = finishedThreads.map(cachedModel.modelGradients(_))
      val pOffset = parameters.paramOffset
      val pLength = parameters.size
      val taskSize = pLength / subModelNum
      val extraSize = pLength % subModelNum
      val parallelNum = if (taskSize == 0) extraSize else subModelNum
      // add gradients from different threads to the first model
      if (parallelNum != 1) {
        Engine.default.invokeAndWait((0 until parallelNum).map(tid => () => {
          val offset = pOffset + tid * taskSize + math.min(tid, extraSize)
          val length = taskSize + (if (tid < extraSize) 1 else 0)
          val mainGradients = finishedGradients.head.narrow(0, offset, length)
          var i = 1
          while (i < finishedGradients.length) {
            mainGradients.add(finishedGradients(i).narrow(0, offset, length))
            i += 1
          }
        }))
      }
      // "finishedGradients" contains all gradients, which will be split to
      // different partitions in putGradients function
      parameters.putGradients(finishedGradients.head)
    } else {
      cachedModel.modelGradients.head.zero()
      parameters.putGradients(cachedModel.modelGradients.head)
    }
  }

  // use gradients to update weights
  private[optim] def updateParameters(
      models: RDD[Cache[Float]],
      parameters: AllReduceParameter[Float],
      driverState: Table): Unit = {

    val stateBroadcast = models.sparkContext.broadcast(driverState)
    // foreachPartition
    models.mapPartitions { modelIter =>
      val localOptimMethod = modelIter.next().optimMethod
      val localState = stateBroadcast.value
      parameters.aggregateGradientPartition(localState[Int]("numFinishedModelUpdates"))

      localOptimMethod.optimize(_ => (localState[Float]("loss"), parameters.gradientPartition),
        parameters.weightPartition)

      localOptimMethod.state.update("epoch", localState[Int]("epoch"))
      localOptimMethod.state.update("trainIter", localState[Int]("trainIter"))
      localOptimMethod.state.update("loss", localState[Double]("loss"))

      parameters.sendWeightPartition()
      Iterator.empty
    }.count()  // use count() to trigger mapPartitions transformation

    stateBroadcast.destroy()
  }

  private def getLatestFile(path: String, fileName: String): String = {
    val fl = new java.io.File(path)
    val files = fl.listFiles(new FilenameFilter {
      override def accept(dir: File, name: String): Boolean = {
        name.startsWith(fileName)
      }
    })

    val last = files.reduce((a, b) => if (a.lastModified() > b.lastModified()) a else b)
    last.getPath
  }
}
