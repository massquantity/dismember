package com.mass.dr.dataset

import com.mass.dr.Path
import com.mass.sparkdl.tensor.Tensor

class MiniBatch(
    itemPathMapping: Map[Int, Seq[Path]],
    numLayer: Int,
    numPathPerItem: Int,
    seqLen: Int,
    val batchSize: Int,
    val originalDataSize: Int) {
  import MiniBatch._

  private var offset: Int = -1
  private var length: Int = -1
  val numTargetsPerBatch: Int = math.max(1, batchSize / numPathPerItem)

  def updatePosition(offset: Int, length: Int): this.type = {
    this.offset = offset
    this.length = length
    this
  }

  def transformLayerData(data: Array[DRSample], offset: Int, length: Int): LayerTransformedBatch = {
    val totalLen = length * numPathPerItem
    val samples = Array.range(offset, offset + length).map(data(_))
    val copiedItems = samples.flatMap(s => Seq.fill(numPathPerItem)(s.sequence).flatten)
    val itemSeqs = Tensor(copiedItems, Array(totalLen, seqLen))

    // An item may contain multiple paths.
    val itemPaths = (1 until numLayer).map { i =>
      // val paths = samples.flatMap(s => itemPathMapping(s.target).flatMap(_.slice(0, i)))
      val _p =
        for {
          s <- samples
          path <- itemPathMapping(s.target)
        } yield path.slice(0, i)
      Tensor(_p.flatten, Array(totalLen, i))
    }

    val targets = (0 until numLayer).map { i =>
      val _t =
        for {
          s <- samples
          path <- itemPathMapping(s.target)
        } yield path(i)
      Tensor(_t, Array(totalLen, 1))
    }
    LayerTransformedBatch(itemSeqs, itemPaths, targets)
  }

  def transformRerankData(data: Array[DRSample], offset: Int, length: Int): RerankTransformedBatch = {
    val samples = data.slice(offset, offset + length)
    val itemSeqs = Tensor(samples.flatMap(_.sequence), Array(length, seqLen))
    val targets = Tensor(samples.map(_.target), Array(length, 1))
    RerankTransformedBatch(itemSeqs, targets)
  }

  def getOffset: Int = offset

  def getLength: Int = length
}

object MiniBatch {

  sealed trait TransformedBatch

  case class LayerTransformedBatch(
    itemSeqs: Tensor[Int],
    paths: IndexedSeq[Tensor[Int]],
    targets: IndexedSeq[Tensor[Int]]) extends TransformedBatch

  case class RerankTransformedBatch(
    itemSeqs: Tensor[Int],
    target: Tensor[Int]) extends TransformedBatch

}
