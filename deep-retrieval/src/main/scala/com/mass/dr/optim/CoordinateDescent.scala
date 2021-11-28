package com.mass.dr.optim

import scala.collection.mutable

import com.mass.dr.dataset.{DRTrainSample, LocalDataSet}
import com.mass.dr.model.Recommender
import com.mass.dr.model.Recommender.PathScore
import com.mass.dr.{LayerModule, Path}

class CoordinateDescent(
    numIteration: Int,
    dataset: LocalDataSet,
    models: Seq[LayerModule[Double]],
    numCandidatePath: Int,
    numPathPerItem: Int,
    decayFactor: Double = 0.999,
    penaltyFactor: Double = 3e-6,
    penaltyPolyOrder: Int = 4) {
  import CoordinateDescent._
  val itemPathMapping = mutable.Map.empty[Int, IndexedSeq[Path]]

  def optimize(): Unit = {
    val allItems = dataset.getData.map(_.target).distinct
    val itemOccurrence = computeItemOccurrence(dataset)
    val itemPathScore = computePathScore(dataset, models, numCandidatePath, decayFactor)
    val pathSize = mutable.Map.empty[Path, Int]
    for {
      t <- 1 to numIteration
      v <- allItems
    } {
      val finalPath = List.range(0, numPathPerItem).foldRight((List.empty[Path], 0.0)) {
        case (j, (selectedPath, partialSum)) =>
          if (t > 1) {
            val lastItemPath = itemPathMapping(v)(j)
            pathSize(lastItemPath) -= 1
          }
          val candidatePath =
            if (selectedPath.isEmpty) {
              itemPathScore(v)
            } else {
              itemPathScore(v).filterNot(n => selectedPath.contains(n.path))
            }
          val incrementalGain = candidatePath.map { n =>
            val size = pathSize.getOrElse(n.path, 0)
            val penalty = penaltyFactor * penaltyFunc(size, penaltyPolyOrder)
            (n.path, itemOccurrence(v) * (math.log(n.prob + partialSum) - math.log(partialSum)) - penalty)
          }
          val (path, score) = incrementalGain.maxBy(_._2)
          pathSize(path) = pathSize.getOrElse(path, 0) + 1
          (path :: selectedPath, partialSum + score)
      }
      itemPathMapping(v) = finalPath._1.toIndexedSeq
    }
  }
}

object CoordinateDescent extends Recommender {

  val penaltyFunc: (Int, Int) => Double = (pathSize, polyOrder) => {
    val _func = (_s: Int) => math.pow(_s, polyOrder) / polyOrder
    _func(pathSize + 1) - _func(pathSize)
  }

  def computeItemOccurrence(dataset: LocalDataSet): Map[Int, Int] = {
    val data = dataset.getData.map(_.target)
    data.groupBy(identity).mapValues(_.length)
  }

  def computePathScore(
      dataset: LocalDataSet,
      models: Seq[LayerModule[Double]],
      numCandidatePath: Int,
      decayFactor: Double = 0.999): mutable.Map[Int, Seq[PathScore]] = {
    val itemPathScores = mutable.Map.empty[Int, Seq[PathScore]]
    val data = dataset.getData
    data.foreach { case DRTrainSample(sequence, item) =>
      val candidatePath = beamSearch(sequence, models, numCandidatePath)
      if (!itemPathScores.contains(item)) {
        itemPathScores(item) = candidatePath
      } else {
        val originalPath = itemPathScores(item)
        val minScore = originalPath.minBy(_.prob).prob
        val originalPathScore = originalPath.view.map(i => i.path -> i.prob).toMap
        val candidatePathScore = candidatePath.view.map(i => i.path -> i.prob).toMap
        val unionPath = originalPathScore.keySet.union(candidatePathScore.keySet).toArray
        val newPath = unionPath.map { p =>
          val newScore =
            if (originalPathScore.contains(p) && candidatePathScore.contains(p)) {
              decayFactor * originalPathScore(p) + candidatePathScore(p)
            } else if (!originalPathScore.contains(p) && candidatePathScore.contains(p)) {
              decayFactor * minScore + candidatePathScore(p)
            } else {
              decayFactor * originalPathScore(p)
            }
          PathScore(p, newScore)
        }
        itemPathScores(item) = newPath.sortBy(_.prob)(Ordering[Double].reverse).take(numCandidatePath)
      }
    }
    itemPathScores
  }
}
