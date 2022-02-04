package com.mass.dr.model

import com.mass.sparkdl.nn.{Embedding, Input, Linear, Reshape}
import com.mass.sparkdl.nn.graphnn.Graph
import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.tensor.TensorNumeric.NumericDouble

class RerankModel(
    numItem: Int,
    seqLen: Int,
    embedSize: Int,
    paddingIdx: Int) {
  import RerankModel.extractCandidates

  private[dr] val model = buildModel()
  lazy val embedParams: Tensor[Double] = model.fetchModuleParameters(
    "embedding", "weight"
  )
  lazy val linearParams: Seq[Tensor[Double]] = model.fetchModuleParameters(
    "linear", Seq("weight", "bias")
  )

  private def buildModel(): Graph[Double] = {
    val flattenSize = seqLen * embedSize
    val inputSeq = Input()
    val embedding = Embedding(numItem, embedSize, paddingIdx)
      .setName("embedding")
      .inputs(inputSeq)
    val embedFlatten = Reshape(Array(flattenSize))
      .setName("flatten")
      .inputs(embedding)
    val linear = Linear(flattenSize, embedSize)
      .setName("linear")
      .inputs(embedFlatten)
    Graph(Array(inputSeq), Array(linear))
  }

  def forward(input: Tensor[Double]): Tensor[Double] = {
    model.forward(input).toTensor
  }

  def backward(input: Tensor[Double], gradOutput: Tensor[Double]): Tensor[Double] = {
    model.backward(input, gradOutput).toTensor
  }

  def inference(
      candidateItems: Seq[Int],
      inputSeq: Seq[Int],
      softmaxWeights: Tensor[Double],
      softmaxBiases: Tensor[Double],
      embedSize: Int): Array[Double] = {
    val output = Tensor[Double](candidateItems.length)
    val userVector = inferenceUserVector(inputSeq, embedSize)
    val candidateWeights = extractCandidates(candidateItems, softmaxWeights, embedSize)
    val candidateBiases = extractCandidates(candidateItems, softmaxBiases)
    output.addmv(0.0, 1.0, candidateWeights, userVector)
    output.add(candidateBiases)
    output.storage().array()
  }

  def inferenceUserVector(inputSeq: Seq[Int], embedSize: Int): Tensor[Double] = {
    val output = Tensor[Double](embedSize)
    val (linearWeight, linearBias) = (linearParams.head, linearParams.last)
    val embedArray = embedParams.storage().array()
    val inputData = inputSeq.flatMap { i =>
      if (i == paddingIdx) {
        Seq.fill(embedSize)(0.0)
      } else {
        embedArray.slice(i * embedSize, i * embedSize + embedSize)
      }
    }
    val inputEmbed = Tensor(inputData.toArray, Array(inputSeq.length * embedSize))
    output.addmv(0.0, 1.0, linearWeight, inputEmbed)
    output.add(linearBias)
    output
  }
}

object RerankModel {

  def apply(
      numItems: Int,
      seqLen: Int,
      embedSize: Int,
      paddingIdx: Int): RerankModel = {
    new RerankModel(
      numItems,
      seqLen,
      embedSize,
      paddingIdx
    )
  }

  private def extractCandidates(
      candidateItems: Seq[Int],
      embedWeights: Tensor[Double],
      stepSize: Int = 1): Tensor[Double] = {
    val weightData = embedWeights.storage().array()
    val dimension = embedWeights.dim()
    dimension match {
      case 1 =>
        val outputData = candidateItems.map(weightData(_))
        Tensor(outputData.toArray, Array(candidateItems.length))
      case _ =>
        val outputData = candidateItems.flatMap { i =>
          weightData.slice(i * stepSize, i * stepSize + stepSize)
        }
        Tensor(outputData.toArray, Array(candidateItems.length, stepSize))
    }
  }
}
