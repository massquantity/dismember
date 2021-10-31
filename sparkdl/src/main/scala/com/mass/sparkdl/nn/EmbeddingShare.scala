package com.mass.sparkdl.nn

import scala.reflect.ClassTag

import com.mass.sparkdl.nn.abstractnn.AbstractModule
import com.mass.sparkdl.nn.mixin.LookupTable
import com.mass.sparkdl.tensor.{Tensor, TensorNumeric}
import com.mass.sparkdl.utils.{T, Table}

class EmbeddingShare[T: ClassTag](
    val nIndex: Int,
    val embedSize: Int,
    val paddingIdx: Int = -1)(
    implicit ev: TensorNumeric[T]) extends AbstractModule[Table, Table, T] with LookupTable[T] {

  var weight: Tensor[T] = Tensor[T](nIndex, embedSize).randn(0.0, 0.05)
  var gradWeight: Tensor[T] = Tensor[T](nIndex, embedSize).zero()

  private val inputBuffer = T()
  override val zeroArray: Array[T] = Array.fill[T](embedSize)(ev.zero)

  padWeight(weight, paddingIdx)

  override def updateOutput(input: Table): Table = {
    var i = 0
    while (i < input.length) {
      inputBuffer(i) = input[Tensor[T]](i).contiguous().asInstanceOf[Tensor[Int]]
      val batchSize = input[Tensor[T]](i).size(0)
      val numElem = inputBuffer[Tensor[T]](i).nElement()
      output(i) = output.getOrElse[Tensor[T]](i, Tensor())
        .resize(Array(numElem, embedSize))

      val index = inputBuffer[Tensor[Int]](i).storage().array()
      val indexOffset = inputBuffer[Tensor[Int]](i).storageOffset()
      val outputData = output[Tensor[T]](i).storage().array()
      val outputOffset = output[Tensor[T]](i).storageOffset()
      val weightData = weight.storage().array()
      val weightOffset = weight.storageOffset()

      embeddingLookup(numElem, index, indexOffset, outputData, outputOffset,
        weightData, weightOffset, embedSize, nIndex, paddingIdx)

      if (input[Tensor[T]](i).dim() == 2) {
        output(i) = output[Tensor[T]](i).view(
          batchSize, input[Tensor[T]](i).size(1), embedSize)
      }
      i += 1
    }
    output
  }

  override def updateGradInput(input: Table, gradOutput: Table): Table = {
    var i = 0
    while (i < input.length) {
      gradInput(i) = gradInput.getOrElse[Tensor[T]](i, Tensor())
        .resizeAs(input(i)).zero()
      i += 1
    }
    gradInput
  }

  override def accGradParameters(input: Table, gradOutput: Table): Unit = {
    require(gradWeight.isContiguous, "EmbeddingShare: gradWeight must be contiguous")
    var i = 0
    while (i < input.length) {
      inputBuffer(i) = input[Tensor[T]](i).contiguous().asInstanceOf[Tensor[Int]]
      val numElem = inputBuffer[Tensor[Int]](i).nElement()
      val inputData: Array[Int] = inputBuffer[Tensor[Int]](i).storage().array()
      val inputOffset: Int = inputBuffer[Tensor[Int]](i).storageOffset()
      val _gradOutput = gradOutput[Tensor[T]](i).contiguous()
      val gradWeightData = gradWeight.storage().array()
      val gradWeightOffset = gradWeight.storageOffset()
      val gradOutputData = _gradOutput.storage().array()
      val gradOutputOffset = _gradOutput.storageOffset()

      updateEmbeddings(numElem, inputData, inputOffset, gradOutputData, gradOutputOffset,
        gradWeightData, gradWeightOffset, embedSize, paddingIdx, scaleW)

      i += 1
    }
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight), Array(this.gradWeight))
  }

  override def clearState(): this.type = {
    super.clearState()
    inputBuffer.clear()
    this
  }
}

object EmbeddingShare {
  def apply[@specialized(Float, Double) T: ClassTag](
      nIndex: Int,
      nOutput: Int,
      paddingIdx: Int = -1)(implicit ev: TensorNumeric[T]): EmbeddingShare[T] = {
    new EmbeddingShare[T](nIndex, nOutput, paddingIdx)
  }
}
