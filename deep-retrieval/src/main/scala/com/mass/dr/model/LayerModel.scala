package com.mass.dr.model

import scala.reflect.ClassTag

import com.mass.sparkdl.tensor.TensorNumeric
import com.mass.sparkdl.Module
import com.mass.sparkdl.nn._

object LayerModel {

  def buildModel[@specialized(Float, Double) T: ClassTag](
      numItems: Int,
      numNodes: Int,
      numLayer: Int,
      seqLen: Int,
      embedSize: Int,
      paddingIdx: Int)(implicit ev: TensorNumeric[T]): IndexedSeq[Module[T]] = {
    require(numLayer >= 2, "number of layers must be at least 2")

    val inputSeq = Input[T]()
    val inputPath = Array.fill(numLayer - 1)(Input[T]())

    val userEmbed = Embedding[T](numItems, embedSize, paddingIdx)
      .inputs(inputSeq)
    val userEmbedFlatten = Reshape[T](Array(seqLen * embedSize))
      .inputs(userEmbed)
    val pathEmbed = EmbeddingShare[T](numNodes * (numLayer - 1), embedSize)
      .inputs(inputPath: _*)

    (0 until numLayer) map { d =>
      if (d == 0) {
        val linear = Linear[T](seqLen * embedSize, numNodes)
          .inputs(userEmbedFlatten)
        Graph[T](Array(inputSeq), Array(linear))
      } else {
        val pathEmbedFlatten = Reshape[T](Array(d * embedSize))
          .inputs(Seq((pathEmbed, d - 1)))
        val concat = Concat[T]()
          .inputs(userEmbedFlatten, pathEmbedFlatten)
        val linear = Linear[T]((seqLen + d) * embedSize, numNodes)
          .inputs(concat)
        Graph[T](Array(inputSeq, inputPath(d - 1)), Array(linear))
      }
      // val softmax = SoftMax[T]()
      //  .inputs(linear)
    }
  }
}
