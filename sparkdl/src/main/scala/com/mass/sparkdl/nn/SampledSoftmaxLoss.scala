package com.mass.sparkdl.nn

import java.util.concurrent.ThreadLocalRandom

import scala.collection.mutable
import scala.reflect.ClassTag

import com.mass.sparkdl.nn.abstractnn.AbstractCriterion
import com.mass.sparkdl.tensor.{Tensor, TensorNumeric}

// https://www.tensorflow.org/api_docs/python/tf/nn/sampled_softmax_loss
// This implementation is different from sampled_softmax_loss in TensorFlow in that
// it samples negative items per positive item, whereas sampled_softmax_loss in TensorFlow
// samples negative items per batch. See SampledSoftmaxLossBatch for an equivalent implementation.
// todo: remove_accidental_hits, seed
class SampledSoftmaxLoss[@specialized(Float, Double) T: ClassTag](
    numSampled: Int,
    numClasses: Int,
    embedSize: Int,
    sampledValues: Option[Array[Int]] = None)(
    implicit ev: TensorNumeric[T]) extends AbstractCriterion[T] {
  import SampledSoftmaxLoss.uniformSampler

  private val crossEntropyLoss = CrossEntropyCriterion[T]()
  private var sampledItems: Array[Int] = _
  private var sampledWeightEmbed: Tensor[T] = _
  private var logitsBuffer: Tensor[T] = _
  private var crossEntropyGrad: Tensor[T] = _
  val weights: Tensor[T] = Tensor[T](numClasses, embedSize).randn(0.0, 0.05)
  val biases: Tensor[T] = Tensor[T](numClasses).zero()
  val gradWeights: Tensor[T] = Tensor[T]()
  val gradBiases: Tensor[T] = Tensor[T]()

  // This target is used as item indices in embeddingLookup
  override def updateOutput(inputVecs: Tensor[T], target: Tensor[T]): T = {
    val batchSize = inputVecs.size(0)
    val labels = target.asInstanceOf[Tensor[Int]].storage().array()
    sampledItems = sampledValues match {
      case Some(s) => s
      case None => uniformSampler(labels, numSampled, numClasses)
    }

    val labelPosition = Tensor[T](batchSize).zero()  // all labels are in the first place.
    val sampledBiasEmbed = embeddingLookup(biases, sampledItems, batchSize)
    sampledWeightEmbed = embeddingLookup(weights, sampledItems, batchSize, embedSize)
    logitsBuffer = linear(inputVecs, sampledWeightEmbed, sampledBiasEmbed)
    output = crossEntropyLoss.updateOutput(logitsBuffer, labelPosition)
    output
  }

  // This target is treated as labelPosition in crossEntropyLoss
  override def backward(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    crossEntropyGrad = crossEntropyLoss.updateGradInput(logitsBuffer, target)
    updateGradInput(input, target)
    updateWeightsAndBiases(input, crossEntropyGrad)
    gradInput
  }

  override def updateGradInput(inputVecs: Tensor[T], target: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(inputVecs).zero()
    linearBackward(crossEntropyGrad, sampledWeightEmbed)
    gradInput
  }

  private def embeddingLookup(
      embedWeights: Tensor[T],
      indices: Array[Int],
      batchSize: Int,
      embedSize: Int = 1): Tensor[T] = {

    val numElem = indices.length
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

  private def linear(
      inputVecs: Tensor[T],
      weights: Tensor[T],
      biases: Tensor[T]): Tensor[T] = {

    val batchSize = inputVecs.size(0)
    val output = Tensor[T](batchSize, numSampled + 1)
    var i = 0
    while (i < batchSize) {
      val vec = inputVecs.select(0, i)
      val w = weights.select(0, i)
      val b = biases.select(0, i)
      val out = output.select(0, i)
      out.addmv(ev.zero, ev.one, vec, w)
      out.add(b)
      i += 1
    }
    output
  }

  private def linearBackward(lossGrad: Tensor[T], weightsEmbed: Tensor[T]): Unit = {
    val batchSize = lossGrad.size(0)
    var i = 0
    while (i < batchSize) {
      val logit = lossGrad.select(0, i)
      val w = weightsEmbed.select(0, i)
      val grad = gradInput.select(0, i)
      grad.addmv(ev.zero, ev.one, w.t(), logit)
      i += 1
    }
  }

  private def updateWeightsAndBiases(inputVecs: Tensor[T], lossGrad: Tensor[T]): Unit = {
    gradWeights.resizeAs(sampledWeightEmbed).zero()
    gradBiases.resizeAs(lossGrad).copy(lossGrad)
    val batchSize = inputVecs.size(0)
    var i = 0
    while (i < batchSize) {
      val gradOut = lossGrad.select(0, i).view(lossGrad.size(1), 1)
      val vec = inputVecs.select(0, i).view(1, inputVecs.size(1))
      val gradWeight = gradWeights.select(0, i)
      gradWeight.addmm(ev.zero, ev.one, gradOut, vec)
      i += 1
    }
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
    val length = numSampled + 1
    val sampledResult = new Array[Int](labels.length * length)
    labels.zipWithIndex.foreach { case (posLabel, i) =>
      var offset = i * length
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
