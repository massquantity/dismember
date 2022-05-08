package com.mass.otm.dataset

import java.io.{BufferedReader, InputStreamReader}

import scala.collection.BitSet
import scala.util.{Random, Using}

import com.mass.otm.{encoding, paddingIdx, upperLog2}
import com.mass.otm.dataset.OTMSample.{OTMEvalSample, OTMTrainSample}
import com.mass.sparkdl.utils.{FileReader => DistFileReader}
import org.apache.commons.lang3.math.NumberUtils

class LocalDataSet(
    dataPath: String,
    seqLen: Int,
    minSeqLen: Int,
    splitRatio: Double,
    mode: String,
    seed: Long) {
  import LocalDataSet._
  require(seqLen > 0 && minSeqLen > 0 && seqLen >= minSeqLen)

  private val initData: Seq[InitSample] = readFile(dataPath)
  lazy val (itemIdMapping, idItemMapping) = initializeMapping(initData, mode, seed)
  lazy val allNodes = getAllNodes(idItemMapping.keys.toSeq)
  lazy val DataInfo(userConsumed, trainData, evalData) = generateData(initData)

  private def generateData(initData: Seq[InitSample]): DataInfo = {
    val groupedSamples = initData.groupBy(_.user).toIndexedSeq.map { case (user, samples) =>
      (user, samples.sortBy(_.timestamp).map(_.item).distinct.map(itemIdMapping(_)))
    }
    val (userConsumed, trainSamples, evalSamples) = groupedSamples.map { case (user, items) =>
      if (items.length <= minSeqLen) {
        (user -> items, Seq.empty[OTMTrainSample], Seq.empty[OTMEvalSample])
      } else if (items.length == minSeqLen + 1) {
        (user -> items, Seq(OTMTrainSample(items.init, items.last)), Seq.empty[OTMEvalSample])
      } else {
        val fullSeq = Seq.fill[Int](seqLen - minSeqLen)(paddingIdx) ++ items
        val splitPoint = math.ceil((items.length - minSeqLen) * splitRatio).toInt
        val _trainSamples = fullSeq
          .take(splitPoint + seqLen)
          .sliding(seqLen + 1)
          .map(s => OTMTrainSample(s.init, s.last))
          .toSeq

        val consumed = items.take(splitPoint + minSeqLen)
        val consumedSet = consumed.toSet
        val (_evalSeq, _labels) = fullSeq.splitAt(splitPoint + seqLen)
        val labels = _labels.filterNot(consumedSet) // remove items appeared in train data
        val _evalSamples =
          if (labels.nonEmpty) {
            val evalSeq = _evalSeq.takeRight(seqLen)
            Seq(OTMEvalSample(evalSeq, labels, user))
          } else {
            Seq.empty[OTMEvalSample]
          }
        (user -> consumed, _trainSamples, _evalSamples)
      }
    }.unzip3

    DataInfo(
      userConsumed.toMap,
      trainSamples.filter(_.nonEmpty).flatten,
      evalSamples.filter(_.nonEmpty).flatten
    )
  }

  def trainSize: Int = trainData.length

  def evalSize: Int = evalData.length

  def numItem: Int = itemIdMapping.size

  def numTreeNode: Int = {
    val leafLevel = upperLog2(numItem)
    (math.pow(2, leafLevel + 1) - 1).toInt
  }
}

object LocalDataSet {

  case class InitSample(user: Int, item: Int, timestamp: Long, category: String)

  case class DataInfo(
    userConsumed: Map[Int, Seq[Int]],
    trainData: IndexedSeq[OTMSample],
    evalData: IndexedSeq[OTMSample]
  )

  def apply(
    dataPath: String,
    seqLen: Int,
    minSeqLen: Int,
    splitRatio: Double,
    mode: String,
    seed: Long
  ): LocalDataSet = {
    new LocalDataSet(
      dataPath,
      seqLen,
      minSeqLen,
      splitRatio,
      mode,
      seed
    )
  }

  def readFile(dataPath: String): Seq[InitSample] = {
    val fileReader = DistFileReader(dataPath)
    val inputStream = fileReader.open()
    val lines = Using.resource(new BufferedReader(new InputStreamReader(inputStream, encoding.name))) {
      reader => Iterator.continually(reader.readLine()).takeWhile(_ != null).toSeq
    }
    lines
      .view
      .map(_.trim.split(","))
      .filter(i => i.length == 5 && NumberUtils.isCreatable(i(0)))
      .map(i => InitSample(i(0).toInt, i(1).toInt, i(3).toLong, i(4)))
      .toSeq
  }

  def initializeMapping(
    samples: Seq[InitSample],
    mode: String,
    seed: Long
  ): (Map[Int, Int], Map[Int, Int]) = {
    val rand = new scala.util.Random(seed)
    val uniqueSamples = samples.distinctBy(_.item)
    val orderedItems = mode match {
      case "random" =>
        val items = uniqueSamples.map(_.item)
        rand.shuffle(items)
      case "category" =>
        uniqueSamples.sortWith { (a, b) =>
          a.category < b.category || (a.category == b.category && a.item < b.item)
        }.map(_.item)
    }
    val sampledIds = sampleRandomLeaves(uniqueSamples.length, rand)
    (orderedItems.zip(sampledIds).toMap, sampledIds.zip(orderedItems).toMap)
  }

  private def sampleRandomLeaves(itemNum: Int, rand: Random): Seq[Int] = {
    val leafLevel = upperLog2(itemNum)
    val leafStart = math.pow(2, leafLevel).toInt - 1
    val leafEnd = leafStart * 2 + 1
    rand
      .shuffle(LazyList.range(leafStart, leafEnd))
      .take(itemNum)
      .sorted
  }

  def getAllNodes(ids: Seq[Int]): BitSet = {
    val leafLevel = upperLog2(ids.length)
    ids.foldLeft(BitSet.empty) { (res, i) =>
      val pathNodes = (1 to leafLevel).scanLeft(i)((a, _) => (a - 1) / 2)
      res ++ pathNodes
    }
  }
}
