package com.mass.dr.model

import scala.reflect.ClassTag

import com.mass.dr.RerankModule
import com.mass.sparkdl.nn.{Embedding, Graph, Input, Linear, Reshape, SoftMax}
import com.mass.sparkdl.tensor.{Tensor, TensorNumeric}

object RerankModel {

  def trainModel[@specialized(Float, Double) T: ClassTag](
      numItems: Int,
      seqLen: Int,
      embedSize: Int,
      paddingIdx: Int)(implicit ev: TensorNumeric[T]): RerankModule[T] = {
    val inputSeq = Input[T]()
    val embedding = Embedding[T](numItems, embedSize, paddingIdx)
      .inputs(inputSeq)
    val embedFlatten = Reshape[T](Array(seqLen * embedSize))
      .inputs(embedding)
    // val linear = Linear[T](seqLen * embedSize, numItems)
    //  .inputs(embedFlatten)
    // val softmax = SoftMax[T]()
    //  .inputs(linear)
    Graph[T](Array(inputSeq), Array(embedFlatten))
  }

  def inference[@specialized(Float, Double) T: ClassTag](
      candidateItems: Seq[Int],
      inputSeq: Seq[Int],
      inputModel: RerankModule[T],
      weights: Tensor[T],
      biases: Tensor[T],
      seqLen: Int,
      embedSize: Int)(implicit ev: TensorNumeric[T]): Tensor[T] = {
    val output = Tensor[T](candidateItems.length)
    val input = Tensor[Int](inputSeq, Array(1, inputSeq.length))
    val userVector = inputModel.forward(input).toTensor.squeeze()
    val candidateWeights = extractCandidates(candidateItems, weights, seqLen * embedSize)
    val candidateBiases = extractCandidates(candidateItems, biases)
    output.addmv(ev.zero, ev.one, candidateWeights, userVector)
    output.add(candidateBiases)
    output
  }

  private def extractCandidates[T: ClassTag](
      candidateItems: Seq[Int],
      embedWeights: Tensor[T],
      stepSize: Int = 1)(implicit ev: TensorNumeric[T]): Tensor[T] = {
    val weightData = embedWeights.storage().array()
    val dimension = embedWeights.dim()
    dimension match {
      case 1 =>
        val outputData = candidateItems.map(weightData(_))
        Tensor[T](outputData.toArray, Array(candidateItems.length))
      case _ =>
        val outputData = candidateItems.flatMap(i =>
          weightData.slice(i * stepSize, i * stepSize + stepSize)
        )
        Tensor[T](outputData.toArray, Array(candidateItems.length, stepSize))
    }
  }
}
