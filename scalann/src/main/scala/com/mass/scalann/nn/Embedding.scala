package com.mass.scalann.nn

import scala.reflect.ClassTag

import com.mass.scalann.nn.abstractnn.TensorModule
import com.mass.scalann.nn.mixin.LookupTable
import com.mass.scalann.tensor.{Tensor, TensorNumeric}

class Embedding[T: ClassTag](
    val nIndex: Int,
    val embedSize: Int,
    val paddingIdx: Int = -1,
    val embedWeight: Option[Tensor[T]] = None
)(implicit ev: TensorNumeric[T])
    extends TensorModule[T]
    with LookupTable[T] {

  val weight: Tensor[T] = embedWeight match {
    case Some(embed) => embed
    case None => Tensor[T](nIndex, embedSize).randn(0.0, 0.05)
  }
  val gradWeight: Tensor[T] = Tensor[T](nIndex, embedSize).zero()

  private var inputBuffer = Tensor[Int]()
  override val zeroArray: Array[T] = Array.fill[T](embedSize)(ev.zero)

  padWeight(weight, paddingIdx)

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    inputBuffer = input.contiguous().asInstanceOf[Tensor[Int]]
    val batchSize = input.size(0)
    val numElem = inputBuffer.nElement()
    output.resize(Array(numElem, embedSize))

    val inputData = inputBuffer.storage().array()
    val inputOffset = inputBuffer.storageOffset()
    val outputData = output.storage().array()
    val outputOffset = output.storageOffset()
    val weightData = weight.storage().array()
    val weightOffset = weight.storageOffset()

    embeddingLookup(
      numElem,
      inputData,
      inputOffset,
      outputData,
      outputOffset,
      weightData,
      weightOffset,
      embedSize,
      nIndex,
      paddingIdx
    )

    if (input.dim() == 2) {
      output = output.view(batchSize, input.size(1), embedSize)
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
    require(gradWeight.isContiguous, "Embedding: gradWeight must be contiguous")
    inputBuffer = input.contiguous().asInstanceOf[Tensor[Int]]
    val numElem = inputBuffer.nElement()
    val inputData: Array[Int] = inputBuffer.storage().array()
    val inputOffset: Int = inputBuffer.storageOffset()
    val _gradOutput = gradOutput.contiguous()
    val gradWeightData = gradWeight.storage().array()
    val gradWeightOffset = gradWeight.storageOffset()
    val gradOutputData = _gradOutput.storage().array()
    val gradOutputOffset = _gradOutput.storageOffset()

    updateEmbeddings(
      numElem,
      inputData,
      inputOffset,
      gradOutputData,
      gradOutputOffset,
      gradWeightData,
      gradWeightOffset,
      embedSize,
      paddingIdx,
      scaleW
    )

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

object Embedding {
  def apply[@specialized(Float, Double) T: ClassTag](
      nIndex: Int,
      nOutput: Int,
      paddingIdx: Int = -1,
      embedWeight: Option[Tensor[T]] = None
  )(implicit ev: TensorNumeric[T]): Embedding[T] = {
    new Embedding[T](nIndex, nOutput, paddingIdx, embedWeight)
  }
}
