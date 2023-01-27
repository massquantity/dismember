package com.mass.scalann.nn

import scala.reflect.ClassTag

import com.mass.scalann.nn.abstractnn.AbstractModule
import com.mass.scalann.tensor.{Tensor, TensorNumeric}
import com.mass.scalann.utils.Table
import com.mass.scalann.Module
import com.mass.scalann.nn.graphnn.Graph

class Attention[T: ClassTag](
    embedSize: Int,
    useScale: Boolean = false,
    linearTransform: Boolean = false
)(implicit ev: TensorNumeric[T])
    extends AbstractModule[Table, Tensor[T], T] {

  private val queryLayer = if (linearTransform) {
    Linear(embedSize, embedSize, withBias = false)
  } else {
    Identity()
  }
  private val keyLayer = if (linearTransform) {
    Linear(embedSize, embedSize, withBias = false)
  } else {
    Identity()
  }
  private val attScoreLayer = MatMul(transB = true)
  private val maskLayer = Mask(useScale, embedSize)
  private val softMaxLayer = SoftMax()
  private val combineLayer = MatMul()
  private val linearLayer = Linear(embedSize, embedSize, withBias = false)

  private val model: Module[T] = {
    val queries = Input()
    val keys = Input()
    val mask = Input()

    val queriesTransform = queryLayer.inputs(queries)
    val keysTransform = keyLayer.inputs(keys)

    val attScore = attScoreLayer.inputs(queriesTransform, keysTransform)
    val maskedScore = maskLayer.inputs(attScore, mask)
    val attProb = softMaxLayer.inputs(maskedScore)
    val combined = combineLayer.inputs(attProb, keysTransform)
    val linear = linearLayer.inputs(combined)
    Graph(Seq(queries, keys, mask), Seq(linear))
  }

  override def updateOutput(input: Table): Tensor[T] = {
    output = model.updateOutput(input).toTensor
    output
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = {
    gradInput = model.updateGradInput(input, gradOutput).toTable
    gradInput
  }

  override def accGradParameters(input: Table, gradOutput: Tensor[T]): Unit = {
    model.accGradParameters(input, gradOutput)
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    model.parameters()
  }

  override def clearState(): Attention.this.type = {
    model.clearState()
    this
  }
}

object Attention {
  def apply[@specialized(Float, Double) T: ClassTag](
      embedSize: Int,
      useScale: Boolean = false,
      linearTransform: Boolean = false
  )(implicit ev: TensorNumeric[T]): Attention[T] = {
    new Attention[T](embedSize, useScale, linearTransform)
  }
}
