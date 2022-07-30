package com.mass.dr.dataset

import com.mass.dr.Path
import com.mass.scalann.tensor.Tensor

class MiniBatch(
    numItem: Int,
    numNode: Int,
    numLayer: Int,
    numPathPerItem: Int,
    seqLen: Int) {
  import MiniBatch._

  private var offset: Int = -1
  private var length: Int = -1

  def updatePosition(offset: Int, length: Int): this.type = {
    this.offset = offset
    this.length = length
    this
  }

  def transformLayerData(
    data: Array[DRSample],
    offset: Int,
    length: Int,
    itemPathMapping: Map[Int, Seq[Path]]
  ): LayerTransformedBatch = {
    val totalLen = length * numPathPerItem
    val samples = Array.range(offset, offset + length).map(data(_))
    val concatInputs = (0 until numLayer).map { layer =>
      val layerData = samples.flatMap { s =>
        itemPathMapping(s.target).flatMap { singlePath =>
          if (layer == 0) {
            s.sequence
          } else {
            val nodeIndices = (0 until layer).map(i => singlePath(i) + numItem + i * numNode)
            s.sequence ++ nodeIndices
          }
        }
      }
      Tensor(layerData, Array(totalLen, seqLen + layer))
    }

    val targets = (0 until numLayer).map { i =>
      val _t =
        for {
          s <- samples
          path <- itemPathMapping(s.target)
        } yield path(i).toDouble
      Tensor(_t, Array(totalLen, 1))
    }

    LayerTransformedBatch(concatInputs, targets)
  }

  def transformRerankData(
    data: Array[DRSample],
    offset: Int,
    length: Int
  ): RerankTransformedBatch = {
    val samples = data.slice(offset, offset + length)
    val itemSeqs = Tensor(samples.flatMap(_.sequence), Array(length, seqLen))
    val targets = Tensor(samples.map(_.target), Array(length, 1))
    RerankTransformedBatch(itemSeqs, targets)
  }

  def getOffset: Int = offset

  def getLength: Int = length
}

object MiniBatch {

  def apply(
    numItem: Int,
    numNode: Int,
    numLayer: Int,
    numPathPerItem: Int,
    seqLen: Int
  ): MiniBatch = {
    new MiniBatch(
      numItem,
      numNode,
      numLayer,
      numPathPerItem,
      seqLen
    )
  }

  sealed trait TransformedBatch extends Product with Serializable

  case class LayerTransformedBatch(
    concatInputs: IndexedSeq[Tensor[Int]],  // itemSeqs + paths
    targets: IndexedSeq[Tensor[Double]]) extends TransformedBatch

  case class RerankTransformedBatch(
    itemSeqs: Tensor[Int],
    target: Tensor[Int]) extends TransformedBatch

}
