package com.mass.sparkdl.nn

import scala.reflect.ClassTag

import com.mass.sparkdl.nn.abstractnn.{AbstractModule, TensorModule}
import com.mass.sparkdl.tensor.{Tensor, TensorNumeric}

class LookupTable[T: ClassTag](val nIndex: Int, val nOutput: Int)(
    implicit ev: TensorNumeric[T]) extends TensorModule[T]{

  var weight: Tensor[T] = Tensor[T](nIndex, nOutput).randn(0.0, 0.05)
  var gradWeight: Tensor[T] = Tensor[T](nIndex, nOutput).zero()
  private var inputBuffer = Tensor[Int]()

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    inputBuffer = input.contiguous().asInstanceOf[Tensor[Int]]
  //  if (input.dim() == 2) {
  //    inputBuffer = inputBuffer.view(inputBuffer.nElement())
  //  }

    val batchSize = input.size(0)
    val embedSize = weight.size(1)
    val numEle = inputBuffer.nElement()
    output.resize(Array(numEle, embedSize))

    val inputData = inputBuffer.storage().array()
    val inputOffset = inputBuffer.storageOffset()
    val outputData = output.storage().array()
    val outputOffset = output.storageOffset()
    val weightData = weight.storage().array()
    val weightOffset = weight.storageOffset()

    try {
      var i = 0
      while (i < numEle) {
        val offset1 = inputData(i + inputOffset) * embedSize + weightOffset
        val offset2 = i * embedSize + outputOffset
        System.arraycopy(weightData, offset1, outputData, offset2, embedSize)

      //  output.select(0, i).copy(weight.select(0, inputBuffer(Array(i))))
        i += 1
      }
      if (input.dim() == 2) {
        output = output.view(batchSize, input.size(1), embedSize)
      }
    } catch {
      case e: IllegalArgumentException =>
        throw new IllegalArgumentException(
          s"""LookupTable updateOutput get exception: "${e.getMessage}"\n""" +
          s"please ensure elements of your input will not exceed $nIndex")

      case e: Exception =>
        throw e
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (!gradInput.isSameSizeAs(input)) {
      gradInput.resizeAs(input).zero()
    }
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    require(gradWeight.isContiguous, "LookupTable: gradWeight must be contiguous")
    inputBuffer = input.contiguous().asInstanceOf[Tensor[Int]]
    val inputData: Array[Int] = inputBuffer.storage().array()
    val inputOffset: Int = inputBuffer.storageOffset()
    val numEle: Int = inputBuffer.nElement()
    val _gradOutput = gradOutput.contiguous()
    val gradWeightData = gradWeight.storage().array()
    val gradWeightOffset = gradWeight.storageOffset()
    val gradOutputData = _gradOutput.storage().array()
    val gradOutputOffset = _gradOutput.storageOffset()
    val embedSize = gradWeight.size(1)

    var i = 0
    while (i < numEle) {
      val offset1 = i * embedSize + gradOutputOffset
      val offset2 = inputData(i + inputOffset) * embedSize + gradWeightOffset
    //  val index = inputData(i + inputOffset)
      ev.axpy(
        embedSize,
        ev.fromType(scaleW),
        gradOutputData,
        offset1,
        1,
        gradWeightData,
        offset2,
        1)
      i += 1
    }
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight), Array(this.gradWeight))
  }

  override def clearState(): this.type = {
    super.clearState()
    inputBuffer.set()
    this
  }
}

object LookupTable {
  def apply[@specialized(Float, Double) T: ClassTag](nIndex: Int, nOutput: Int)(
      implicit ev: TensorNumeric[T]): LookupTable[T] = {
    new LookupTable[T](nIndex, nOutput)
  }
}
