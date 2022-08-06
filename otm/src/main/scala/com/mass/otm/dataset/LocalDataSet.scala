package com.mass.otm.dataset

import java.io.{BufferedReader, InputStreamReader}
import java.nio.file.{Files, Paths}

import scala.collection.BitSet
import scala.util.{Random, Using}

import com.mass.otm.{encoding, paddingIdx, upperLog2}
import com.mass.scalann.utils.{FileReader => DistFileReader}
import com.mass.tdm.utils.Serialization.loadBothMapping
import org.apache.commons.lang3.math.NumberUtils

class LocalDataSet(
    dataPath: String,
    seqLen: Int,
    minSeqLen: Int,
    splitRatio: Double,
    leafInitMode: String,
    initMapping: Boolean,
    mappingPath: String,
    labelNum: Int,
    seed: Long,
    dataMode: String) {
  import LocalDataSet._
  require(seqLen > 0 && minSeqLen > 0 && seqLen >= minSeqLen)

  private val rand = new Random(seed)
  private val initData: Array[InitSample] = readFile(dataPath)

  // if mapping doesn't exist, initialize randomly
  lazy val (itemIdMapping, idItemMapping) =
    if (initMapping || !Files.isRegularFile(Paths.get(mappingPath))) {
      initializeMapping(initData, leafInitMode, rand)
    } else {
      println(s"load mapping from $mappingPath")
      loadBothMapping(mappingPath)
    }
  lazy val allNodes = getAllNodes(idItemMapping.keys.toSeq)
  lazy val DataInfo(userConsumed, trainData, evalData) = dataMode match {
    case "one_user_sample" => generateOneSamplePerUser(initData)
    case _ => generateSamples(initData)
  }

  private def generateOneSamplePerUser(initData: Array[InitSample]): DataInfo = {
    val groupedSamples = initData.groupBy(_.user).toSeq.map { case (user, samples) =>
      (user, samples.sortBy(_.timestamp).map(_.item).distinct.map(itemIdMapping(_)))
    }
    val (userConsumed, samples) = groupedSamples
      .foldRight[(Map[Int, Array[Int]], List[OTMSample])]((Map.empty, Nil)) {
        case ((user, items), (userConsumed, samples)) =>
          if (items.length > seqLen) {
            val (sequence, labels) = items.splitAt(seqLen)
            val newSample = OTMSample(sequence, labels.toList, user)
            (userConsumed + (user -> sequence), newSample :: samples)
          } else {
            (userConsumed, samples)
          }
      }
    val splitPoint = (groupedSamples.length * splitRatio).toInt
    val (trainSamples, evalSamples) = rand.shuffle(samples).splitAt(splitPoint)
    DataInfo(userConsumed, trainSamples, evalSamples)  // evalSamples.map(i => i.copy(labels = i.labels.take(10)))
  }

  private def generateSamples(initData: Array[InitSample]): DataInfo = {
    val groupedSamples = initData.groupBy(_.user).toArray.map { case (user, samples) =>
      (user, samples.sortBy(_.timestamp).map(_.item).distinct.map(itemIdMapping(_)))
    }
    val paddingSeq = Array.fill(seqLen - minSeqLen)(paddingIdx)
    groupedSamples.foldRight(DataInfo(Map.empty, Nil, Nil)) {
      case ((user, items), DataInfo(userConsumed, trainSamples, evalSamples)) =>
        if (items.length <= minSeqLen) {
          DataInfo(userConsumed, trainSamples, evalSamples)
        } else if (items.length == minSeqLen + labelNum) {
          val fullSeq = paddingSeq ++: items.take(minSeqLen)
          val newTrainSample = OTMSample(fullSeq, items.drop(minSeqLen).toList, user)
          DataInfo(
            userConsumed + (user -> items),
            newTrainSample :: trainSamples,
            evalSamples
          )
        } else {
          val fullSeq = paddingSeq ++: items
          val splitPoint = math.ceil((items.length - minSeqLen) * splitRatio).toInt
          val seqTrainSamples = fullSeq
            .take(splitPoint + seqLen)
            .sliding(seqLen + labelNum)
            .map(s => OTMSample(s.take(seqLen), s.drop(seqLen).toList, user))
            .toList
          val consumed = items.take(splitPoint + minSeqLen)
          val (evalSeq, labels) = fullSeq.splitAt(splitPoint + seqLen)
          val newEvalSample = OTMSample(evalSeq.takeRight(seqLen), labels.toList, user)
          DataInfo(
            userConsumed + (user -> consumed),
            seqTrainSamples ::: trainSamples,
            newEvalSample :: evalSamples
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
    userConsumed: Map[Int, Array[Int]],
    trainData: List[OTMSample],
    evalData: List[OTMSample]
  )

  def apply(
    dataPath: String,
    seqLen: Int,
    minSeqLen: Int,
    splitRatio: Double,
    leafInitMode: String,
    initMapping: Boolean,
    mappingPath: String,
    labelNum: Int,
    seed: Long,
    dataMode: String = "default"
  ): LocalDataSet = {
    new LocalDataSet(
      dataPath,
      seqLen,
      minSeqLen,
      splitRatio,
      leafInitMode,
      initMapping,
      mappingPath,
      labelNum,
      seed,
      dataMode
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
    samples: Array[InitSample],
    leafInitMode: String,
    rand: Random
  ): (Map[Int, Int], Map[Int, Int]) = {
    val uniqueSamples = samples.distinctBy(_.item)
    val orderedItems = leafInitMode match {
      case "random" =>
        val items = uniqueSamples.map(_.item)
        rand.shuffle(items).toArray
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
