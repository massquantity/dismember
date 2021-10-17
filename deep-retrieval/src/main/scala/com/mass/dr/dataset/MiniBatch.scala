package com.mass.dr.dataset

import com.mass.dr.Path
import com.mass.sparkdl.tensor.Tensor

class MiniBatch(
    itemPathMapping: Map[Int, Seq[Path]],
    numLayer: Int,
    numPathPerItem: Int,
    batchSize: Int,
    seqLen: Int,
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

  def convert(
      data: Array[DRSample],
      threadOffset: Int,
      threadLen: Int): (Feature, Tensor[Int]) = {

    val totalLen = threadLen * numPathPerItem
    val samples = (threadOffset until threadOffset + threadLen).map(data(_)).toArray
    val copedItems = samples.flatMap(s => Array.fill(numPathPerItem)(s.sequence).flatten)
    val itemSeqs = Tensor(copedItems, Array(totalLen, seqLen))

    val itemPaths = (1 until numLayer).map { i =>
      // val paths = samples.flatMap(s => itemPathMapping(s.target).flatMap(_.slice(0, i)))
      val paths =
        for {
          s <- samples
          path <- itemPathMapping(s.target)
        } yield path.slice(0, i)
      Tensor(paths.flatten, Array(totalLen, i))
    }

    val copedTargets = samples.flatMap(s => Array.fill(numPathPerItem)(s.target))
    val targets = Tensor(copedTargets, Array(totalLen, 1))

    (Feature(itemSeqs, itemPaths), targets)
  }
}

object MiniBatch {

  case class Feature(itemSeqs: Tensor[Int], paths: Seq[Tensor[Int]])

}
