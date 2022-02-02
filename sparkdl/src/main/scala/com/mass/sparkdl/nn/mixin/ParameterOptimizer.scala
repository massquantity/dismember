package com.mass.sparkdl.nn.mixin

import scala.reflect.ClassTag

import com.mass.sparkdl.tensor.{Tensor, TensorNumeric}
import com.mass.sparkdl.utils.Table

// use Adam optimizer to update weights and biases in SampledSoftmaxLoss
trait ParameterOptimizer[T] {

  val numClasses: Int
  val embedSize: Int
  val learningRate: Double
  val beta1: Double = 0.9
  val beta2: Double = 0.999
  val epsilon: Double = 1e-7
  val state: Table
  private var timestep = 1
  @transient protected val optimizerBuffer: Tensor[T]

  protected var sampledItems: Array[Int]
  protected var sampledWeightEmbed: Tensor[T]
  protected val weights: Tensor[T]
  protected val biases: Tensor[T]
  protected val gradWeights: Tensor[T]
  protected val gradBiases: Tensor[T]

  protected def updateParameters(inputVecs: Tensor[T], lossGrad: Tensor[T])(
      implicit ev: TensorNumeric[T]): Unit = {
    computeParameterGrad(inputVecs, lossGrad)
    optimize(weights, gradWeights, "weight", timestep)
    optimize(biases, gradBiases, "bias", timestep)
    timestep += 1
  }

  private def optimize(
      parameter: Tensor[T],
      gradient: Tensor[T],
      name: String,
      timestep: Int)(implicit ev: TensorNumeric[T]): Unit = {
    val vName = name + "_v"
    val hName = name + "_h"
    val denomName = name + "_denom"
    val v = state.get[Tensor[T]](vName).get
    val h = state.get[Tensor[T]](hName).get
    val denom = state.get[Tensor[T]](denomName).get

    v.mul(ev.fromType[Double](beta1)).add(ev.fromType[Double](1 - beta1), gradient)
    optimizerBuffer.resizeAs(gradient).cmul(gradient, gradient)
    h.mul(ev.fromType[Double](beta2)).add(ev.fromType[Double](1 - beta2), optimizerBuffer)
    denom.sqrt(h)
    optimizerBuffer.fill(ev.one)
    denom.add(ev.fromType[Double](epsilon), optimizerBuffer)

    val biasCorrection1 = 1 - math.pow(beta1, timestep)
    val biasCorrection2 = 1 - math.pow(beta2, timestep)
    val stepSize = learningRate * math.sqrt(biasCorrection2) / biasCorrection1
    denom.cdiv(v, denom)
    parameter.add(ev.fromType[Double](-stepSize), denom)

    state(vName) = v
    state(hName) = h
    state(denomName) = denom
  }

  private def computeParameterGrad(inputVecs: Tensor[T], lossGrad: Tensor[T])(
      implicit ev: TensorNumeric[T]): Unit = {
    val (weightGradInput, biasGradInput) = computeParameterGradInput(inputVecs, lossGrad)
    val gradWeightsData = gradWeights.storage().array()
    val gradBiasesData = gradBiases.storage().array()
    val numElem = sampledItems.length
    var i = 0
    while (i < numElem) {
      val index = sampledItems(i)
      ev.axpy(embedSize, ev.one, weightGradInput, i * embedSize, 1, gradWeightsData, index * embedSize, 1)
      gradBiasesData(index) = ev.plus(biasGradInput(i), gradBiasesData(index))
      i += 1
    }
  }

  protected def computeParameterGradInput(inputVecs: Tensor[T], lossGrad: Tensor[T]): (Array[T], Array[T])

  def clearOptimizerState(): Unit = {
    state.clear()
  }
}

object ParameterOptimizer {

  def initState[T: ClassTag](
    weights: Tensor[T],
    biases: Tensor[T])(
    implicit ev: TensorNumeric[T]
  ): Table = {
    val state = Table()
    state("weight_v") = Tensor[T]().resizeAs(weights).zero()
    state("weight_h") = Tensor[T]().resizeAs(weights).zero()
    state("weight_denom") = Tensor[T]().resizeAs(weights).zero()
    state("bias_v") = Tensor[T]().resizeAs(biases).zero()
    state("bias_h") = Tensor[T]().resizeAs(biases).zero()
    state("bias_denom") = Tensor[T]().resizeAs(biases).zero()
    state
  }

  def createTensor[T: ClassTag](
    size: Array[Int])(
    implicit ev: TensorNumeric[T]
  ): Tensor[T] = {
    Tensor[T](size).zero()
  }
}
