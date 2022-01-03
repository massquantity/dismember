package com.mass.sparkdl.nn

import java.util.concurrent.ThreadLocalRandom

import scala.collection.mutable
import scala.reflect.ClassTag

import com.mass.sparkdl.nn.abstractnn.AbstractCriterion
import com.mass.sparkdl.nn.mixin.ParameterOptimizer
import com.mass.sparkdl.tensor.{Tensor, TensorNumeric}
import com.mass.sparkdl.utils.Table

// https://www.tensorflow.org/api_docs/python/tf/nn/sampled_softmax_loss
// This implementation is different from `sampled_softmax_loss` in TensorFlow in that
// it samples negative items per positive item, whereas `sampled_softmax_loss` in TensorFlow
// samples negative items per batch. See `SampledSoftmaxLossBatch` for an equivalent implementation.
class SampledSoftmaxLoss[@specialized(Float, Double) T: ClassTag](
    numSampled: Int,
    override val numClasses: Int,
    override val embedSize: Int,
    override val learningRate: Double,
    override val weights: Tensor[T],
    override val biases: Tensor[T],
    sampledValues: Option[Array[Int]] = None)(
    implicit ev: TensorNumeric[T]) extends AbstractCriterion[T] with ParameterOptimizer[T] {
  require(numSampled < numClasses,
    s"numSampled $numSampled < numClasses $numClasses, try using Softmax directly.")
  import SampledSoftmaxLoss.uniformSampler
  import ParameterOptimizer.{initState, createTensor}

  private val crossEntropyLoss = CrossEntropyCriterion[T]()
  private var crossEntropyGrad: Tensor[T] = _
  override protected var sampledItems: Array[Int] = _
  override protected var sampledWeightEmbed: Tensor[T] = _
  private var logitsBuffer: Tensor[T] = _
  private var labelPosition: Tensor[T] = _
  override val gradWeights: Tensor[T] = createTensor(weights.size())
  override val gradBiases: Tensor[T] = createTensor(biases.size())
  override val state: Table = initState(weights, biases)
  override val optimizerBuffer: Tensor[T] = Tensor[T]()

  // This target is used as item indices in embeddingLookup
  override def updateOutput(inputVecs: Tensor[T], target: Tensor[T]): T = {
    val batchSize = inputVecs.size(0)
    val labels = target.toTensor[Int].storage().array()
    sampledItems = sampledValues match {
      case Some(s) => s
      case None => uniformSampler(labels, numSampled, numClasses)  // positive + negative items
    }

    val sampledBiasEmbed = embeddingLookup(biases, sampledItems, batchSize)
    sampledWeightEmbed = embeddingLookup(weights, sampledItems, batchSize, embedSize)
    logitsBuffer = linear(inputVecs, sampledWeightEmbed, sampledBiasEmbed)
    labelPosition = Tensor[T](batchSize).zero()  // all labels are in the first place.
    output = crossEntropyLoss.updateOutput(logitsBuffer, labelPosition)
    output
  }

  // This target is useless, the real target is labelPosition.
  override def backward(inputVecs: Tensor[T], target: Tensor[T]): Tensor[T] = {
    crossEntropyGrad = crossEntropyLoss.updateGradInput(logitsBuffer, labelPosition)
    updateGradInput(inputVecs, target)
    updateParameters(inputVecs, crossEntropyGrad)
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
    embedSize: Int = 1
  ): Tensor[T] = {
    val weightData = embedWeights.storage().array()
    if (embedWeights.dim() == 1) {
      val outputData = indices.map(weightData(_))
      Tensor[T](outputData, Array(batchSize, numSampled + 1))
    } else {
      val outputData = new Array[T](indices.length * embedSize)
      for ((index, j) <- indices.zipWithIndex) {
        val weightOffset = index * embedSize
        val outputOffset = j * embedSize
        System.arraycopy(weightData, weightOffset, outputData, outputOffset, embedSize)
      }
      Tensor[T](outputData, Array(batchSize, numSampled + 1, embedSize))
    }
  }

  private def linear(
      inputVecs: Tensor[T],
      weights: Tensor[T],
      biases: Tensor[T]): Tensor[T] = {
    val batchSize = inputVecs.size(0)
    val output = Tensor[T](batchSize, numSampled + 1)
    0 until batchSize foreach { i =>
      val vec = inputVecs.select(0, i)
      val w = weights.select(0, i)
      val b = biases.select(0, i)
      val out = output.select(0, i)
      out.addmv(ev.zero, ev.one, w, vec)
      out.add(b)
    }
    output
  }

  private def linearBackward(lossGrad: Tensor[T], weightsEmbed: Tensor[T]): Unit = {
    val batchSize = lossGrad.size(0)
    0 until batchSize foreach { i =>
      val logit = lossGrad.select(0, i)
      val w = weightsEmbed.select(0, i)
      val grad = gradInput.select(0, i)
      grad.addmv(ev.zero, ev.one, w.t(), logit)
    }
  }

  override def computeParameterGradInput(
      inputVecs: Tensor[T],
      lossGrad: Tensor[T]
  ): (Array[T], Array[T]) = {
    val weightsGradInput = Tensor[T]().resizeAs(sampledWeightEmbed).zero()
    val biasesGradInput = Tensor[T]().resizeAs(lossGrad).copy(lossGrad)
    val batchSize = inputVecs.size(0)
    0 until batchSize foreach { i =>
      val gradOut = lossGrad.select(0, i).view(lossGrad.size(1), 1)
      val vec = inputVecs.select(0, i).view(1, inputVecs.size(1))
      val gradWeight = weightsGradInput.select(0, i)
      gradWeight.addmm(ev.zero, ev.one, gradOut, vec)
    }
    (weightsGradInput.storage().array(), biasesGradInput.storage().array())
  }
}

object SampledSoftmaxLoss {

  def apply[@specialized(Float, Double) T: ClassTag](
      numSampled: Int,
      numClasses: Int,
      embedSize: Int,
      learningRate: Double,
      weights: Tensor[T],
      biases: Tensor[T],
      sampledValues: Option[Array[Int]] = None)(
      implicit ev: TensorNumeric[T]): SampledSoftmaxLoss[T] = {
    new SampledSoftmaxLoss[T](
      numSampled,
      numClasses,
      embedSize,
      learningRate,
      weights,
      biases,
      sampledValues
    )
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
