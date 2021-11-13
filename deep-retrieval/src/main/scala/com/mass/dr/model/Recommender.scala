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
    val topNodes = beamSearch(sequence, models, beamSize)
    for {
      node <- topNodes
      item <- pathItemsMapping(node.path)
      if pathItemsMapping.contains(node.path)
    } yield item
  }

  def beamSearch(
      sequence: Array[Int],
      models: Seq[LayerModule[Double]],
      beamSize: Int): Seq[Node] = {
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

  case class Node(path: Path, prob: Double)

  val firstLayerProbs: (LayerModule[Double], Tensor[Int], Int) => Array[Node] = (
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
      .map(i => Node(IndexedSeq(i._2), i._1))
  }

  val restLayerProbs: (LayerModule[Double], Tensor[Int], Node) => Array[Node] = (
    model: LayerModule[Double],
    itemSeqs: Tensor[Int],
    node: Node,
  ) => {
      val inputPath = Tensor[Int](node.path.toArray, Array(1, node.path.length))
      val modelInput = T(itemSeqs, inputPath)
      model.forward(modelInput)
        .toTensor
        .storage()
        .array()
        .zipWithIndex
        .map(p => Node(node.path ++ Seq(p._2), p._1 * node.prob))
  }
}
