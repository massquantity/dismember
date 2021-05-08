package com.mass.sparkdl.dataset

import scala.reflect.ClassTag

import com.mass.sparkdl.tensor.{Tensor, TensorNumeric}

trait MiniBatch[T] extends Serializable {

  def size(): Int

  def slice(offset: Int, length: Int): MiniBatch[T]

  def getInput: Tensor[T]

  def getTarget: Tensor[T]

  def set(samples: Seq[Sample[T]])(implicit ev: TensorNumeric[T]): this.type
}

private[sparkdl] class ArrayTensorMiniBatch[T: ClassTag](
    val inputData: Tensor[T],
    val targetData: Tensor[T]) extends MiniBatch[T] {

  protected var batchSize = 0
  protected var unlabeled = false

  override def size(): Int = inputData.size(0)

  override def slice(offset: Int, length: Int): MiniBatch[T] = {
    val inputs = inputData.narrow(0, offset, length)
    val targets = targetData.narrow(0, offset, length)
    MiniBatch(inputs, targets)
  }

  override def getInput: Tensor[T] = inputData

  override def getTarget: Tensor[T] = targetData

  override def set(samples: Seq[Sample[T]])(implicit ev: TensorNumeric[T]): this.type = {
    require(samples.nonEmpty, "samples is empty")
    require(batchSize == 0 || samples.length <= batchSize)
    val resize = batchSize != samples.length || size() != samples.length

    if (batchSize == 0) {
      batchSize = samples.length
    }
    if (resize) {
      MiniBatch.resize(samples, this)
    }
    MiniBatch.copy(samples, this)
    this
  }
}

object MiniBatch {

  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): MiniBatch[T] = {
    new ArrayTensorMiniBatch[T](Tensor[T](), Tensor[T]())
  }

  def apply[T: ClassTag](input: Tensor[T], target: Tensor[T]): MiniBatch[T] = {
    new ArrayTensorMiniBatch[T](input, target)
  }

  def apply[T: ClassTag](input: Tensor[T]): MiniBatch[T] = {
    new ArrayTensorMiniBatch[T](input, null)
  }

  private[sparkdl] def resize[T: ClassTag](
      samples: Seq[Sample[T]], miniBatch: ArrayTensorMiniBatch[T])(
      implicit ev: TensorNumeric[T]): MiniBatch[T] = {
    val inputs = miniBatch.inputData
    val targets = miniBatch.targetData

    val featureSizes = Array(samples.length) ++ samples.head.getFeatureSize
    inputs.resize(featureSizes)
    if (samples.head.labelLength() != 0) {
      val labelSizes = Array(samples.length) ++ samples.head.getLabelSize
      targets.resize(labelSizes)
    }
    miniBatch
  }

  private[sparkdl] def copy[T: ClassTag](
      samples: Seq[Sample[T]], miniBatch: ArrayTensorMiniBatch[T])(
      implicit ev: TensorNumeric[T]): MiniBatch[T] = {

    val inputs = miniBatch.inputData
    val targets = miniBatch.targetData
    val hasLabel = samples.head.labelLength() != 0

    var i = 0
    while (i < samples.length) {
      val sample = samples(i)
      val sampleData = sample.getData
      val sampleLen = sample.getFeatureSize.product
      var offset = 0
      copy(sampleData, offset, inputs(i), sampleLen)
      offset += sampleLen

      if (hasLabel) {
        val targetLen = sample.getLabelSize.product
        copy(sampleData, offset, targets(i), targetLen)
        // offset += targetLen
      }

      i += 1
    }
    miniBatch
  }

  private def copy[T: ClassTag](src: Array[T], offset: Int, dest: Tensor[T], length: Int)(
      implicit ev: TensorNumeric[T]): Unit = {
    ev.arraycopy(src, offset, dest.storage().array(), dest.storageOffset(), length)
  }
}
