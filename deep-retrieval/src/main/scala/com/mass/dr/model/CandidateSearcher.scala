package com.mass.dr.model

import com.mass.dr.Path

trait CandidateSearcher {
  import CandidateSearcher.{PathScore, PathInfo}

  def searchCandidate(
    inputSeq: Seq[Int],
    models: LayerModel,
    beamSize: Int,
    pathItemsMapping: Map[Path, Seq[Int]]
  ): Seq[Int] = {
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
    beamSize: Int
  ): Seq[PathScore] = {
    val initValue = Seq(
      PathInfo(
        intermediateInput = sequence,
        intermediatePath = Nil,
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
