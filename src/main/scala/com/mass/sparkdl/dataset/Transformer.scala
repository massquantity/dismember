package com.mass.sparkdl.dataset

import scala.reflect.ClassTag

import com.mass.sparkdl.tensor.TensorNumeric
import org.apache.commons.lang3.SerializationUtils
import org.apache.spark.rdd.RDD

trait Transformer[A, B] extends Serializable {

  def apply(prev: Iterator[A]): Iterator[B]

  // def -> [C](other: Transformer[B, C]): Transformer[A, C] = {
  //  new ChainedTransformer(this, other)
  // }

  def cloneTransformer(): Transformer[A, B] = {
    SerializationUtils.clone(this)
  }

  def apply(dataset: RDD[A])(implicit ev: ClassTag[B]): RDD[B] = {
    val broadcast = dataset.sparkContext.broadcast(this)
    val cachedTransformer = dataset.mapPartitions(_ =>
      Iterator.single(broadcast.value.cloneTransformer())
    ).setName("Transformer")

    dataset.zipPartitions(cachedTransformer)((data, tran) => tran.next()(data))
  }
}

/* class ChainedTransformer[A, B, C](
    first: Transformer[A, B],
    last: Transformer[B, C]) extends Transformer[A, C] {

  override def apply(prev: Iterator[A]): Iterator[C] = {
    last(first(prev))
  }
} */

class SampleToMiniBatch[T: ClassTag](
    totalBatch: Int,
    miniBatch: Option[MiniBatch[T]] = None,
    partitionNum: Option[Int] = None)(
    implicit ev: TensorNumeric[T]) extends Transformer[Sample[T], MiniBatch[T]] {

  var miniBatchBuffer: MiniBatch[T] = miniBatch.orNull
  private val batchSize = DataUtil.getBatchSize(totalBatch, partitionNum)
  private val sampleData = new Array[Sample[T]](batchSize)

  override def apply(prev: Iterator[Sample[T]]): Iterator[MiniBatch[T]] = {

    new Iterator[MiniBatch[T]] {

      override def hasNext: Boolean = prev.hasNext

      override def next(): MiniBatch[T] = {
        if (prev.hasNext) {
          var i = 0
          while (i < batchSize && prev.hasNext) {
            sampleData(i) = prev.next()
            i += 1
          }

          if (null == miniBatchBuffer) {
            miniBatchBuffer = MiniBatch()
          }

          if (i < batchSize) {
            miniBatchBuffer.set(sampleData.slice(0, i))
          } else {
            miniBatchBuffer.set(sampleData)
          }
        } else {
          null
        }
      }
    }
  }
}

object SampleToMiniBatch {
  def apply[T: ClassTag](batchSize: Int, partitionNum: Option[Int] = None)(
      implicit ev: TensorNumeric[T]): SampleToMiniBatch[T] = {
    new SampleToMiniBatch[T](batchSize, None, partitionNum)
  }

  def apply[T: ClassTag](miniBatch: MiniBatch[T], batchSize : Int, partitionNum: Option[Int])(
      implicit ev: TensorNumeric[T]): SampleToMiniBatch[T] = {
    new SampleToMiniBatch[T](batchSize, Some(miniBatch), partitionNum)
  }
}
