package com.mass.otm.dataset

import java.io.{BufferedReader, InputStreamReader}

import scala.collection.BitSet
import scala.util.{Random, Using}

import com.mass.otm.{encoding, paddingIdx, upperLog2}
import com.mass.sparkdl.utils.{FileReader => DistFileReader}
import org.apache.commons.lang3.math.NumberUtils

class LocalDataSet(
    dataPath: String,
    seqLen: Int,
    minSeqLen: Int,
    splitRatio: Double,
    leafInitMode: String,
    labelMode: String,
    seed: Long) {
  import LocalDataSet._
  require(seqLen > 0 && minSeqLen > 0 && seqLen >= minSeqLen)
  // println(f"\tEval time: ${(System.nanoTime() - start) / 1e9d}%.4fs, ")

  private val rand = new Random(seed)
  private val initData: Array[InitSample] = readFile(dataPath)
  lazy val (itemIdMapping, idItemMapping) = initializeMapping(initData, leafInitMode, rand)
  lazy val allNodes = getAllNodes(idItemMapping.keys.toSeq)
  lazy val DataInfo(userConsumed, trainData, evalData) = labelMode match {
    case "singleLabel" => generateWithSingleLabel(initData)
    case "multiLabel" => generateWithMultiLabel(initData)
  }

  private def generateWithMultiLabel(initData: Array[InitSample]): DataInfo = {
    val groupedSamples = initData.groupBy(_.user).toSeq.map { case (user, samples) =>
      (user, samples.sortBy(_.timestamp).map(_.item).distinct.map(itemIdMapping(_)))
    }
    val (userConsumed, samples) = groupedSamples
      .foldRight[(Map[Int, Seq[Int]], List[OTMSample])]((Map.empty, Nil)) {
        case ((user, items), (userConsumed, samples)) =>
          if (items.length > seqLen) {
            val (sequence, labels) = items.splitAt(seqLen)
            (userConsumed + (user -> sequence), OTMSample(sequence, labels.toList, user) :: samples)
          } else {
            (userConsumed, samples)
          }
      }
    val splitPoint = (groupedSamples.length * splitRatio).toInt
    val (trainSamples, evalSamples) = rand.shuffle(samples).splitAt(splitPoint)
    DataInfo(userConsumed, trainSamples, evalSamples)  // evalSamples.map(i => i.copy(labels = i.labels.take(10)))
  }

  private def generateWithSingleLabel(initData: Array[InitSample]): DataInfo = {
    val groupedSamples = initData.groupBy(_.user).toArray.map { case (user, samples) =>
      (user, samples.sortBy(_.timestamp).map(_.item).distinct.map(itemIdMapping(_)))
    }
    groupedSamples.foldRight(DataInfo(Map.empty, Nil, Nil)) {
      case ((user, items), DataInfo(userConsumed, trainSamples, evalSamples)) =>
        if (items.length <= minSeqLen) {
          DataInfo(userConsumed, trainSamples, evalSamples)
        } else if (items.length == minSeqLen + 1) {
          DataInfo(
            userConsumed + (user -> items),
            OTMSample(items.init, List(items.last), user) :: trainSamples,
            evalSamples
          )
        } else {
          val fullSeq = Array.fill[Int](seqLen - minSeqLen)(paddingIdx) ++: items
          val splitPoint = math.ceil((items.length - minSeqLen) * splitRatio).toInt
          val seqTrain = fullSeq
            .take(splitPoint + seqLen)
            .sliding(seqLen + 1)
            .map(s => OTMSample(s.init, List(s.last), user))
            .toList
          val consumed = items.take(splitPoint + minSeqLen)
          val (evalSeq, labels) = fullSeq.splitAt(splitPoint + seqLen)
          DataInfo(
            userConsumed + (user -> consumed),
            seqTrain ::: trainSamples,
            OTMSample(evalSeq.takeRight(seqLen), labels.toList, user) :: evalSamples
          )
        }
    }
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
    trainData: List[OTMSample],
    evalData: List[OTMSample]
  )

  def apply(
    dataPath: String,
    seqLen: Int,
    minSeqLen: Int,
    splitRatio: Double,
    leafInitMode: String,
    labelMode: String,
    seed: Long
  ): LocalDataSet = {
    new LocalDataSet(
      dataPath,
      seqLen,
      minSeqLen,
      splitRatio,
      leafInitMode,
      labelMode,
      seed
    )
  }

  def readFile(dataPath: String): Array[InitSample] = {
    val fileReader = DistFileReader(dataPath)
    val inputStream = fileReader.open()
    val lines = Using.resource(new BufferedReader(new InputStreamReader(inputStream, encoding.name))) {
      reader => Iterator.continually(reader.readLine()).takeWhile(_ != null).toArray
    }
    lines
      .view
      .map(_.trim.split(","))
      .filter(i => i.length == 5 && NumberUtils.isCreatable(i(0)))
      .map(i => InitSample(i(0).toInt, i(1).toInt, i(3).toLong, i(4)))
      .toArray
  }

  def initializeMapping(
    samples: Seq[InitSample],
    leafInitMode: String,
    rand: Random
  ): (Map[Int, Int], Map[Int, Int]) = {
    val uniqueSamples = samples.distinctBy(_.item)
    val orderedItems = leafInitMode match {
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
