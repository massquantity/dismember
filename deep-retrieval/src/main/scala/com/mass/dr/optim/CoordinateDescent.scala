package com.mass.dr.optim

import scala.collection.mutable
import scala.util.Random

import com.mass.sparkdl.utils.Engine
import com.mass.dr.dataset.LocalDataSet
import com.mass.dr.model.{CandidateSearcher, LayerModel}
import com.mass.dr.model.CandidateSearcher.PathScore
import com.mass.dr.{Path => DRPath}

class CoordinateDescent(
    dataset: LocalDataSet,
    batchSize: Int,
    numIteration: Int,
    numCandidatePath: Int,
    numPathPerItem: Int,
    numLayer: Int,
    numNode: Int,
    decayFactor: Double = 0.999,
    penaltyFactor: Double = 3e-6,
    penaltyPolyOrder: Int = 4) {
  import CoordinateDescent._
  val numThread = Engine.coreNumber()
  val itemOccurrence = computeItemOccurrence(dataset)

  def optimize(model: LayerModel, trainMode: String): Map[Int, Seq[DRPath]] = {
    val itemPathScore = trainMode match {
      case "batch" => batchPathScore(
        dataset,
        model,
        numCandidatePath,
        batchSize,
        numThread
      )
      case "streaming" => streamingPathScore(
        dataset,
        model,
        numCandidatePath,
        decayFactor,
        batchSize,
        numThread
      )
    }
    val itemPathMapping = mutable.Map.empty[Int, Seq[DRPath]]
    val pathSize = mutable.Map.empty[DRPath, Int]
    for {
      t <- 1 to numIteration
      v <- dataset.idItemMapping.keys
    } {
      itemPathMapping(v) = if (itemOccurrence.contains(v)) {
        List.range(0, numPathPerItem).foldRight[(List[DRPath], Double)]((Nil, 1e-5)) {
          case (j, (selectedPath, partialSum)) =>
            if (t > 1) {
              val lastItemPath = itemPathMapping(v)(j)
              pathSize(lastItemPath) -= 1
            }
            val candidatePath = selectedPath match {
              case Nil => itemPathScore(v)
              case _ => itemPathScore(v).filterNot(n => selectedPath.contains(n.path))
            }
            val incrementalGain = candidatePath.map { n =>
              val size = pathSize.getOrElse(n.path, 0)
              val penalty = penaltyFactor * penaltyFunc(size, penaltyPolyOrder)
              val nv = itemOccurrence(v)
              (n.path, nv * (math.log(n.prob + partialSum) - math.log(partialSum)) - penalty)
            }
            val (maxPath, score) = incrementalGain.maxBy(_._2)
            pathSize(maxPath) = pathSize.getOrElse(maxPath, 0) + 1
            (maxPath :: selectedPath, partialSum + score)
        }._1
      } else {
        generateRandomPath(numPathPerItem, numLayer, numNode)
      }
    }
    itemPathMapping.toMap
  }
}

object CoordinateDescent extends CandidateSearcher {

  def apply(
    dataset: LocalDataSet,
    batchSize: Int,
    numIteration: Int,
    numCandidatePath: Int,
    numPathPerItem: Int,
    numLayer: Int,
    numNode: Int,
    decayFactor: Double,
    penaltyFactor: Double,
    penaltyPolyOrder: Int
  ): CoordinateDescent = {
    new CoordinateDescent(
      dataset,
      batchSize,
      numIteration,
      numCandidatePath,
      numPathPerItem,
      numLayer,
      numNode,
      decayFactor,
      penaltyFactor,
      penaltyPolyOrder
    )
  }

  val penaltyFunc: (Int, Int) => Double = (pathSize, polyOrder) => {
    val _func = (_s: Int) => math.pow(_s, polyOrder) / polyOrder
    _func(pathSize + 1) - _func(pathSize)
  }

  val aggregatePathScore: Int => Seq[Seq[PathScore]] => Seq[PathScore] = num => pathScores =>
    pathScores
      .flatten
      .groupMapReduce(_.path)(_.prob)(_ + _)
      .toSeq
      .sortBy(_._2)(Ordering[Double].reverse)
      .take(num)
      .map(i => PathScore(i._1, i._2))

  def computeItemOccurrence(dataset: LocalDataSet): Map[Int, Int] = {
    val items = dataset.getTrainData.map(_.target)
    items.groupBy(identity).view.mapValues(_.length).toMap
  }

  def batchPathScore(
    dataset: LocalDataSet,
    model: LayerModel,
    numCandidatePath: Int,
    batchSize: Int,
    numThread: Int
  ): Map[Int, Seq[PathScore]] = {
    dataset.getTrainData.sliding(batchSize, batchSize).toSeq.flatMap { batchData =>
      val threadDataSize = math.ceil(batchData.length.toDouble / numThread).toInt
      Engine.default.invokeAndWait(
        batchData.sliding(threadDataSize, threadDataSize).toSeq.map { threadData => () =>
          threadData.map { d =>
            val candidatePath = beamSearch(d.sequence, model, numCandidatePath)
            (d.target, candidatePath)
          }
        }
      ).flatten
    }.groupMap(_._1)(_._2)
      .view
      .mapValues(aggregatePathScore(numCandidatePath))
      .toMap
  }

  def streamingPathScore(
    dataset: LocalDataSet,
    model: LayerModel,
    numCandidatePath: Int,
    decayFactor: Double,
    batchSize: Int,
    numThread: Int
  ): Map[Int, Seq[PathScore]] = {
    val itemPathScores = mutable.Map.empty[Int, Seq[PathScore]]
    val data = dataset.getTrainData
    for {
      batchData <- data.sliding(batchSize, batchSize).toSeq
      threadDataSize = math.ceil(batchData.length.toDouble / numThread).toInt
      threadData = batchData.sliding(threadDataSize, threadDataSize).toSeq
    } {
      Engine.default.invokeAndWait(threadData.map { td => () =>
        td.map { d =>
          val candidatePath = beamSearch(d.sequence, model, numCandidatePath)
          (d.target, candidatePath)
        }
      }).flatten
        .foreach { case (item, candidatePath) =>
          if (!itemPathScores.contains(item)) {
            itemPathScores(item) = candidatePath
          } else {
            val originalPath = itemPathScores(item)
            val minScore = originalPath.minBy(_.prob).prob
            val originalPathScore = originalPath.view.map(i => i.path -> i.prob).toMap
            val candidatePathScore = candidatePath.view.map(i => i.path -> i.prob).toMap
            val unionPath = originalPathScore.keySet.union(candidatePathScore.keySet).toSeq
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
            itemPathScores(item) = newPath
              .sortBy(_.prob)(Ordering[Double].reverse)
              .take(numCandidatePath)
          }
        }
    }
    Map.empty ++ itemPathScores
  }

  def generateRandomPath(numPathPerItem: Int, numLayer: Int, numNode: Int): Seq[DRPath] = {
    for {
      _ <- 1 to numPathPerItem
    } yield (1 to numLayer).map(_ => Random.nextInt(numNode))
  }
}
