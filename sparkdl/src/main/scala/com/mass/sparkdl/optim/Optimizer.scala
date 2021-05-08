package com.mass.sparkdl.optim

import scala.reflect.{classTag, ClassTag}

import com.mass.sparkdl.tensor.TensorNumeric
import com.mass.sparkdl.{Criterion, DataSet, Module}
import com.mass.sparkdl.dataset._
import com.mass.sparkdl.utils.{T, Table}
import org.apache.spark.rdd.RDD

abstract class Optimizer[T: ClassTag, D](
    protected var model: Module[T],
    protected var dataset: DataSet[D],
    protected var criterion: Criterion[T])(implicit ev: TensorNumeric[T]) {

  protected var state: Table = T()
  protected var optimMethods: Map[String, OptimMethod[T]] = Map(model.getName -> new Adam())
  protected var endWhen: Trigger = Trigger.maxIteration(100, "trainIter")

  protected var dropPercentage: Double = 0.0
  protected var maxDropPercentage: Double = 0.0
  protected var computeThresholdbatchSize: Int = 100
  protected var warmupIterationNum: Int = 200

  def optimize(): Module[T]



  private[optim] def shutdown(): Unit = { }

  def reserveOptim(reserve: Boolean): this.type = {
    throw new UnsupportedOperationException(
      "Only support DistriOptimizer to reserve optim methods for each worker")
  }
}

object Optimizer {

  private[sparkdl] def header(epoch: Int, count: Int, total: Long, iter: Int,
      wallClockTime: Long): String = {
    s"[Epoch $epoch $count/$total][Iteration $iter][Wall Clock ${wallClockTime / 1e9}s]"
  }

  def apply[T: ClassTag](
      model: Module[T],
      sampleRDD: RDD[Sample[T]],
      criterion: Criterion[T],
      batchSize: Int)(
      implicit ev: TensorNumeric[T]): Optimizer[T, MiniBatch[T]] = {

    new DistriOptimizer[T](
      _model = model,
      _dataset = (DataSet.rdd(sampleRDD) -> SampleToMiniBatch(batchSize)).toDistributed,
      _criterion = criterion
    )
  }

  def apply[T: ClassTag, D](
      model: Module[T],
      dataset: DataSet[D],
      criterion: Criterion[T])(
      implicit ev: TensorNumeric[T]): Optimizer[T, D] = {

    dataset match {
      case _: DistributedDataSet[_] =>
        new DistriOptimizer[T](
          _model = model,
          _dataset = dataset.asInstanceOf[DistributedDataSet[MiniBatch[T]]],
          _criterion = criterion
        ).asInstanceOf[Optimizer[T, D]]

      /* case _: LocalDataSet[_] =>
        new LocalOptimizer[T](
          model = model,
          dataset = dataset.toLocal.asInstanceOf[LocalDataSet[MiniBatch[T]]],
          criterion = criterion
        ).asInstanceOf[Optimizer[T, D]] */

      case _ => throw new UnsupportedOperationException
    }
  }
}
