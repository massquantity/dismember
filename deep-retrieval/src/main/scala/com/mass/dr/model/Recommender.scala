package com.mass.dr.model

import scala.collection.mutable.ArrayBuffer

import com.mass.dr.Path
import com.mass.sparkdl.Module
import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.utils.T

trait Recommender {
  import com.mass.sparkdl.tensor.TensorNumeric.NumericDouble
  import Recommender.Node

  val pathProbs: (Module[Double], Node, Tensor[Int]) => Array[Node] = (
    model: Module[Double],
    node: Node,
    itemSeqs: Tensor[Int],
  ) => {
    val inputPath = Tensor[Int](node.path, Array(1, node.path.length))
    val modelInput = T(itemSeqs, inputPath)
    model.forward(modelInput)
      .toTensor
      .storage()
      .array()
      .zipWithIndex
      .map(p => Node(node.path ++ IndexedSeq(p._2), p._1 * node.prob))
  }

  def recommend(
      sequence: Array[Int],
      models: IndexedSeq[Module[Double]],
      topk: Int,
      beamSize: Int,
      // mappingPi: Map[Int, Path],
      pathItemsMapping: Map[Path, Seq[Int]],   // path -> items on this path
      consumedItems: Option[Array[Int]] = None): Seq[Int] = {

    implicit val ord: Ordering[Double] = Ordering[Double].reverse
    val itemSeqs = Tensor[Int](sequence, Array(1, sequence.length))
    val firstLayer = models.head.forward(itemSeqs)
      .toTensor
      .storage()
      .array()
      .zipWithIndex
      .sortBy(_._1)
      .take(beamSize)
      .map(i => Node(IndexedSeq(i._2), i._1))
    val probOutputs = ArrayBuffer[Node](firstLayer: _*)

    var i = 1
    while (i < models.length) {
      val candidatePaths = probOutputs.flatMap(node => pathProbs(models(i), node, itemSeqs))
      probOutputs.clear()
      probOutputs ++= candidatePaths.sortBy(_.prob).take(beamSize)
      i += 1
    }

    probOutputs
      .flatMap(n => pathItemsMapping.getOrElse(n.path, Seq(-1)))
      .filterNot(_ == -1)
  }
}

object Recommender {

  case class Node(path: Path, prob: Double)

}
