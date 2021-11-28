package com.mass.dr.model

import com.mass.dr.{LayerModule, Path}
import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.tensor.TensorNumeric.NumericDouble
import com.mass.sparkdl.utils.T

trait Recommender {
  import Recommender._

  def recommend(
      sequence: Array[Int],
      models: Seq[LayerModule[Double]],
      beamSize: Int,
      pathItemsMapping: Map[Path, Seq[Int]]): Seq[Int] = {
    val topPaths = beamSearch(sequence, models, beamSize).map(_.path)
    for {
      path <- topPaths
      item <- pathItemsMapping(path)
      if pathItemsMapping.contains(path)
    } yield item
  }

  def beamSearch(
      sequence: Array[Int],
      models: Seq[LayerModule[Double]],
      beamSize: Int): Seq[PathScore] = {
    val itemSeqs = Tensor[Int](sequence, Array(1, sequence.length))
    val firstLayerNodes = firstLayerProbs(models.head, itemSeqs, beamSize)
    models.tail.foldLeft(firstLayerNodes) { (nodes, model) =>
      nodes
        .flatMap(n => restLayerProbs(model, itemSeqs, n))
        .sortBy(_.prob)
        .take(beamSize)
    }
  }
}

object Recommender {

  implicit val ord: Ordering[Double] = Ordering[Double].reverse

  case class PathScore(path: Path, prob: Double)

  val firstLayerProbs: (LayerModule[Double], Tensor[Int], Int) => Array[PathScore] = (
    model: LayerModule[Double],
    itemSeqs: Tensor[Int],
    beamSize: Int
  ) => {
    model.forward(itemSeqs)
      .toTensor
      .storage()
      .array()
      .zipWithIndex
      .sortBy(_._1)
      .take(beamSize)
      .map(i => PathScore(IndexedSeq(i._2), i._1))
  }

  val restLayerProbs: (LayerModule[Double], Tensor[Int], PathScore) => Array[PathScore] = (
    model: LayerModule[Double],
    itemSeqs: Tensor[Int],
    node: PathScore,
  ) => {
      val inputPath = Tensor[Int](node.path.toArray, Array(1, node.path.length))
      val modelInput = T(itemSeqs, inputPath)
      model.forward(modelInput)
        .toTensor
        .storage()
        .array()
        .zipWithIndex
        .map(p => PathScore(node.path ++ Seq(p._2), p._1 * node.prob))
  }
}
