package com.mass.dr.model

import com.mass.dr.{LayerModule, Path}
import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.tensor.TensorNumeric.NumericDouble
import com.mass.sparkdl.utils.Table

trait CandidateSearcher {
  import CandidateSearcher.{PathScore, PathInfo}

  def searchCandidate(
      inputSeq: Seq[Int],
      models: LayerModel,
      beamSize: Int,
      pathItemsMapping: Map[Path, Seq[Int]]): Seq[Int] = {
    val topPaths = beamSearch(inputSeq, models, beamSize).map(_.path)
    for {
      path <- topPaths
      if pathItemsMapping.contains(path)
      item <- pathItemsMapping(path)
    } yield item
  }

  def beamSearch(
      sequence: Seq[Int],
      model: LayerModel,
      beamSize: Int): Seq[PathScore] = {
    val initValue = Seq(
      PathInfo(
        intermediateInput = sequence,
        intermediatePath = List.empty[Int],
        candidateNode = -1,
        probability = 1.0)
    )
    val paths = Seq.range(0, model.numLayer).foldLeft(initValue) { (pathInfos, i) =>
      val offset = model.numItem + i * model.numNode
      val candidatePaths = pathInfos.flatMap { pi =>
        model.inference(pi.intermediateInput, i)
          .zipWithIndex
          .map { case (prob, node) =>
            PathInfo(
              pi.intermediateInput,
              pi.intermediatePath,
              node,
              pi.probability * prob
            )
          }
      }
      candidatePaths
        .sortBy(_.probability)(Ordering[Double].reverse)
        .take(beamSize)
        .map { pi =>
          pi.copy(
            intermediateInput = pi.intermediateInput :+ (pi.candidateNode + offset),
            intermediatePath = pi.candidateNode :: pi.intermediatePath
          )
        }
    }
    paths.map(p => PathScore(p.intermediatePath.reverse.toIndexedSeq, p.probability))
  }

  private val firstLayerProbs: (LayerModule[Double], Tensor[Int], Int) => Array[PathScore] = (
    model: LayerModule[Double],
    itemSeqs: Tensor[Int],
    beamSize: Int
  ) => {
    model.forward(itemSeqs)
      .toTensor
      .storage()
      .array()
      .take(100)  // take certain size since the underlying array may be larger than u think
      .zipWithIndex
      .sortBy(_._1)(Ordering[Double].reverse)
      .take(beamSize)
      .map(i => PathScore(IndexedSeq(i._2), i._1))
  }

  private val restLayerProbs: (LayerModule[Double], Tensor[Int], PathScore) => Array[PathScore] = (
    model: LayerModule[Double],
    itemSeqs: Tensor[Int],
    node: PathScore,
  ) => {
    val inputPath = Tensor[Int](node.path.toArray, Array(1, node.path.length))
    val modelInput = Table(itemSeqs, inputPath)
    model.forward(modelInput)
      .toTensor
      .storage()
      .array()
      .take(100)
      .zipWithIndex
      .map(p => PathScore(node.path :+ p._2, p._1 * node.prob))
  }

  // implicit val ord: Ordering[Double] = Ordering[Double].reverse

  // def beamSearch(
  //  sequence: Seq[Int],
  //  models: Seq[LayerModule[Double]],
  //  beamSize: Int): Seq[PathScore] = {
  //  val itemSeqs = Tensor[Int](sequence.toArray, Array(1, sequence.length))
  //  val firstLayerNodes = firstLayerProbs(models.head, itemSeqs, beamSize)
  //  models.tail.foldLeft(firstLayerNodes) { (nodes, model) =>
  //    nodes
  //      .flatMap(n => restLayerProbs(model, itemSeqs, n))
  //      .sortBy(_.prob)(Ordering[Double].reverse)
  //      .take(beamSize)
  //  }
  // }
}

object CandidateSearcher {

  case class PathScore(path: Path, prob: Double)

  case class PathInfo(
    intermediateInput: Seq[Int],
    intermediatePath: List[Int],
    candidateNode: Int,
    probability: Double
  )
}
