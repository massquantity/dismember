package com.mass.scalann.nn

import scala.reflect.ClassTag

import com.mass.scalann.nn.abstractnn.AbstractCriterion
import com.mass.scalann.tensor.{Tensor, TensorNumeric}

// implementation based on...
// https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Loss.cpp
// binary_cross_entropy_with_logits
class BCECriterionWithLogits[@specialized(Float, Double) T: ClassTag](sizeAverage: Boolean = true)(
    implicit ev: TensorNumeric[T]
) extends AbstractCriterion[T] {

  private var buffer: Array[T] = _
  private var buffer2: Array[T] = _
  private var ones: Array[T] = _

  def checkSize(inputSize: Array[Int], targetSize: Array[Int]): Boolean = {
    var input = List[Int]()
    var target = List[Int]()
    inputSize.foreach(i => if (i != 1) input = i :: input)
    targetSize.foreach(i => if (i != 1) target = i :: target)
    input == target
  }

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    checkSize(input.size(), target.size())

    output = ev.fromType[Int](0)
    val inputData = input.storage().array()
    val targetData = target.storage().array()
    val batchSize = input.nElement()
    val iOffset = input.storageOffset()
    val tOffset = target.storageOffset()
    if (buffer == null || buffer.length < batchSize) buffer = new Array[T](batchSize)
    if (buffer2 == null || buffer2.length < batchSize) buffer2 = new Array[T](batchSize)
    if (ones == null || ones.length < batchSize) ones = Array.fill[T](batchSize)(ev.one)

    // compute max(x, 0) - x * z + log(1 + exp(-abs(x)))
    var i = 0
    while (i < batchSize) {
      buffer(i) = ev.max(inputData(iOffset + i), ev.zero)
      i += 1
    }
    ev.abs(batchSize, inputData, iOffset, buffer2, 0)
    ev.scal(batchSize, ev.fromType(-1.0), buffer2, 0, 1)
    ev.vExp(batchSize, buffer2, 0, buffer2, 0)
    ev.vAdd(batchSize, buffer2, 0, ones, 0, buffer2, 0)
    // ev.axpy(batchSize, ev.fromType(1.0), ones, 0, 1, buffer2, 0, 1)
    ev.vLn(batchSize, buffer2, 0, buffer2, 0)
    ev.vAdd(batchSize, buffer, 0, buffer2, 0, buffer, 0)
    // ev.axpy(batchSize, ev.fromType(1.0), buffer2, 0, 1, buffer, 0, 1)
    output = ev.minus(
      ev.dot(batchSize, buffer, 0, 1, ones, 0, 1),
      ev.dot(batchSize, inputData, iOffset, 1, targetData, tOffset, 1)
    )

    if (sizeAverage) {
      output = ev.divide(output, ev.fromType(batchSize))
    }
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input).copy(input)
    val gradInData = gradInput.storage().array()
    val targetData = target.storage().array()
    val batchSize = input.nElement()
    val offset = gradInput.storageOffset()
    if (ones == null) ones = Array.fill[T](batchSize)(ev.one)

    // compute sigmoid(input) - y
    ev.scal(batchSize, ev.fromType(-1.0), gradInData, offset, 1)
    ev.vExp(batchSize, gradInData, offset, gradInData, offset)
    ev.vAdd(batchSize, gradInData, offset, ones, 0, gradInData, offset)
    // inv = pow^(-1)
    ev.vPowx(batchSize, gradInData, offset, ev.fromType(-1.0), gradInData, offset)
    ev.vSub(batchSize, gradInData, offset, targetData, target.storageOffset(), gradInData, offset)

    if (sizeAverage) {
      gradInput.mul(ev.fromType(1.0 / batchSize))
    }
    gradInput
  }
}

object BCECriterionWithLogits {
  def apply[@specialized(Float, Double) T: ClassTag](
      sizeAverage: Boolean = true
  )(implicit ev: TensorNumeric[T]): BCECriterionWithLogits[T] = {
    new BCECriterionWithLogits[T](sizeAverage)
  }
}
