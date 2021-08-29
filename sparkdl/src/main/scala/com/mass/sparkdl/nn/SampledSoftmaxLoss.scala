package com.mass.sparkdl.nn

import java.util.concurrent.ThreadLocalRandom

import scala.collection.mutable
import scala.reflect.ClassTag

import com.mass.sparkdl.nn.abstractnn.AbstractCriterion
import com.mass.sparkdl.tensor.{Tensor, TensorNumeric}

// An implementation based upon sampled_softmax_loss in TensorFlow.
// https://www.tensorflow.org/api_docs/python/tf/nn/sampled_softmax_loss
class SampledSoftmaxLoss[@specialized(Float, Double) T: ClassTag](
    numSampled: Int,
    numClasses: Int,
    embedSize: Int)(implicit ev: TensorNumeric[T]) extends AbstractCriterion[T] {
  import SampledSoftmaxLoss.uniformSampler

  private var sampledBuffer: Array[Int] = _
  private var logitsBuffer: Tensor[T] = _
  private var weightsBuffer: Tensor[T] = _
  private var gradBuffer: Tensor[T] = _
  val weights: Tensor[T] = Tensor[T](numClasses, embedSize).randn(0.0, 0.05)
  val biases: Tensor[T] = Tensor[T](numClasses).zero()
  val gradWeights: Tensor[T] = Tensor[T](numClasses, embedSize).zero()
  val gradBiases: Tensor[T] = Tensor[T](numClasses).zero()
  private val loss = CrossEntropyCriterion[T]()

  override def backward(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    updateGradInput(input, target)
    updateEmbeddings()
    gradInput
  }

  override def updateOutput(inputVecs: Tensor[T], target: Tensor[T]): T = {
    val batchSize = inputVecs.size(0)
    val labels = target.asInstanceOf[Tensor[Int]].storage().array()
    sampledBuffer = uniformSampler(labels, numSampled, numClasses)
    weightsBuffer = embeddingLookup(weights, sampledBuffer, batchSize, embedSize)
    val biasesEmbed = embeddingLookup(biases, sampledBuffer, batchSize)
    logitsBuffer = linear(inputVecs, weightsBuffer, biasesEmbed)
    val labelPosition = Tensor[T](batchSize).zero()
    loss.updateOutput(logitsBuffer, labelPosition)
  }

  override def updateGradInput(inputVecs: Tensor[T], labelPosition: Tensor[T]): Tensor[T] = {
    gradInput = loss.updateGradInput(logitsBuffer, labelPosition)
    gradBuffer = linearBackward(inputVecs)
    gradInput
  }

  private def embeddingLookup(
      embedWeights: Tensor[T],
      indices: Array[Int],
      batchSize: Int,
      embedSize: Int = 1): Tensor[T] = {

    val numElem = indices.length
    // val sampleNum = numElem / batchSize
    val outputData = new Array[T](numElem * embedSize)
    val weightData = embedWeights.storage().array()
    var i = 0
    while (i < numElem) {
      val index = indices(i)
      if (embedWeights.dim() == 1) {
        outputData(i) = weightData(index)
      } else {
        val weightOffset = index * embedSize
        val outputOffset = i * embedSize
        System.arraycopy(weightData, weightOffset, outputData, outputOffset, embedSize)
      }
      i += 1
    }

    if (embedWeights.dim() == 1) {
      Tensor[T](outputData, Array(batchSize, numSampled + 1))
    } else {
      Tensor[T](outputData, Array(batchSize, numSampled + 1, embedSize))
    }
  }

  private def updateEmbeddings(): Unit = {
    val numElem = sampledBuffer.length
    val gradData = gradBuffer.storage().array()
    val gradInputData = gradInput.storage().array()
    val gradWeightData = gradWeights.storage().array()
    val gradBiasData = gradBiases.storage().array()
    var i = 0
    while (i < numElem) {
      val index = sampledBuffer(i)
      val gradOffset = i * embedSize
      val gradWeightOffset = index * embedSize
      ev.axpy(
        embedSize,
        ev.one,
        gradData,
        gradOffset,
        1,
        gradWeightData,
        gradWeightOffset,
        1)
      gradBiasData(index) = gradInputData(i)
      i += 1
    }
  }

  private def linear(
      inputVecs: Tensor[T],
      weights: Tensor[T],
      biases: Tensor[T]): Tensor[T] = {

    val batchSize = inputVecs.size(0)
    // val sampleNum = weights.size(1)
    val output = Tensor[T](Array(batchSize, numSampled + 1))
    var i = 0
    while (i < batchSize) {
      val vec = inputVecs.select(0, i)
      val w = weights.select(0, i)
      val b = biases.select(0, i)
      val out = output.select(0, i)
      out.addmv(ev.one, vec, w)
      out.add(b)
      i += 1
    }
    output
  }

  private def linearBackward(inputVecs: Tensor[T]): Tensor[T] = {
    val batchSize = inputVecs.size(0)
    var i = 0
    while (i < batchSize) {
      val logit = logitsBuffer.select(0, i)
      val w = weightsBuffer.select(0, i)
      val grad = gradInput.select(0, i)
      grad.addmv(ev.one, logit, w.t())
      i += 1
    }
    val gradBuffer = Tensor[T](Array(numSampled + 1, embedSize))
    gradBuffer.addmm(ev.one, gradInput.t(), inputVecs)
    gradBuffer
  }
}

object SampledSoftmaxLoss {

  def apply[@specialized(Float, Double) T: ClassTag](
      numSampled: Int,
      numClasses: Int,
      embedSize: Int)(implicit ev: TensorNumeric[T]): SampledSoftmaxLoss[T] = {
    new SampledSoftmaxLoss[T](numSampled, numClasses, embedSize)
  }

  private def uniformSampler(labels: Array[Int], numSampled: Int, numClasses: Int): Array[Int] = {
    val oneLen = numSampled + 1
    val sampledResult = new Array[Int](labels.length * oneLen)
    labels.zipWithIndex.foreach { case (posLabel, i) =>
      var offset = i * oneLen
      sampledResult(offset) = posLabel
      offset += 1
      val hasSampled = new mutable.BitSet()
      while (hasSampled.size < numSampled) {
        val s = ThreadLocalRandom.current.nextInt(numClasses)
        if (!hasSampled.contains(s) && s != posLabel) {
          hasSampled += s
        }
      }
      hasSampled.foreach { s =>
        sampledResult(offset) = s
        offset += 1
      }
    }
    sampledResult
  }
}
