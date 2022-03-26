package com.mass.otm.dataset

import java.io.{BufferedReader, InputStreamReader}

import scala.collection.mutable
import scala.util.Using

import com.mass.otm.dataset.OTMSample.{OTMEvalSample, OTMTrainSample}
import com.mass.otm.encoding
import com.mass.sparkdl.dataset.DataUtil
import com.mass.sparkdl.utils.{FileReader => DistFileReader}
import org.apache.commons.lang3.math.NumberUtils

class LocalDataSet(
    dataPath: String,
    totalTrainBatchSize: Int,
    totalEvalBatchSize: Int,
    seqLen: Int,
    minSeqLen: Int,
    startSampleLevel: Int = 1,
    splitRatio: Double = 0.8,
    numThreads: Int = 1,
    useMask: Boolean,
    paddingIdx: Int = -1) {
  import LocalDataSet._
  require(seqLen > 0 && minSeqLen > 0 && seqLen >= minSeqLen)
  require(startSampleLevel > 0, s"start sample level should be at least 1, got $startSampleLevel")

  private val trainBatchSize = totalTrainBatchSize
  private val evalBatchSize = totalEvalBatchSize
  private val initData: Array[InitSample] = readFile(dataPath)

  lazy val (itemIdMapping, itemPathMapping) = initializeMapping(initData)
  lazy val (userConsumed, trainData, evalData) = generateData()

  private def generateData(): (Map[Int, Array[Int]], Array[OTMTrainSample], Array[OTMEvalSample]) = {
    val groupedSamples = initData.groupBy(_.user).toArray.map { case (user, samples) =>
      (user, samples.sortBy(_.timestamp).map(_.item).distinct.map(itemIdMapping(_)))
    }
    val (userConsumed, trainSamples, evalSamples) = groupedSamples.map { case (user, items) =>
      if (items.length <= minSeqLen) {
        (user -> items, Array.empty[OTMTrainSample], Array.empty[OTMEvalSample])
      } else if (items.length == minSeqLen + 1) {
        (user -> items, Array(OTMTrainSample(items.init, items.last)), Array.empty[OTMEvalSample])
      } else {
        val fullSeq = Array.fill[Int](seqLen - minSeqLen)(paddingIdx) ++ items
        val splitPoint = math.ceil((items.length - minSeqLen) * splitRatio).toInt
        val _trainSamples = fullSeq
          .take(splitPoint + seqLen)
          .sliding(seqLen + 1)
          .map(s => OTMTrainSample(s.init, s.last))
          .toArray

        val consumed = items.take(splitPoint + minSeqLen)
        val consumedSet = consumed.toSet
        val (_evalSeq, _labels) = fullSeq.splitAt(splitPoint + seqLen)
        val labels = _labels.filterNot(consumedSet) // remove items appeared in train data
        val _evalSamples =
          if (labels.nonEmpty) {
            val evalSeq = _evalSeq.takeRight(seqLen)
            Array(OTMEvalSample(evalSeq, labels, user))
          } else {
            Array.empty[OTMEvalSample]
          }
        (user -> consumed, _trainSamples, _evalSamples)
      }
    }.unzip3

    val validTrain = trainSamples.filter(_.nonEmpty).flatten
    val validEval = evalSamples.filter(_.nonEmpty).flatten
    (userConsumed.toMap, validTrain, validEval)
  }

  def shuffle(): Unit = DataUtil.shuffle(trainData)

  def trainSize: Int = trainData.length

  def evalSize: Int = evalData.length

  def numItem: Int = itemIdMapping.size

}

object LocalDataSet {

  case class InitSample(user: Int, item: Int, timestamp: Long)

  private def readFile(dataPath: String): Array[InitSample] = {
    val fileReader = DistFileReader(dataPath)
    val inputStream = fileReader.open()
    val lines = Using.resource(new BufferedReader(new InputStreamReader(inputStream, encoding.name))) {
      reader => Iterator.continually(reader.readLine()).takeWhile(_ != null).toArray
    }
    lines
      .map(_.trim.split(","))
      .filter(i => i.length == 5 && NumberUtils.isCreatable(i(0)))
      .map(i => InitSample(i(0).toInt, i(1).toInt, i(3).toLong))
  }

  private def initializeMapping(data: Array[InitSample]): (Map[Int, Int], Map[Int, Int]) = {
    val items = data.map(_.item).distinct
    val shuffledItems = scala.util.Random.shuffle(items)
    val sampledIds = sampleRandomLeaves(items)
    (shuffledItems.zip(sampledIds).toMap, sampledIds.zip(shuffledItems).toMap)
  }

  private def sampleRandomLeaves(items: Array[Int]): Seq[Int] = {
    val log2 = (n: Int) => math.ceil(math.log(n) / math.log(2)).toInt
    val leafLevel = log2(items.length)
    val start = math.pow(2, leafLevel).toInt - 1
    val end = start * 2 + 1
    val hasSampled = mutable.BitSet.empty
    while (hasSampled.size < items.length) {
      val s = scala.util.Random.between(start, end)
      if (!hasSampled.contains(s)) {
        hasSampled += s
      }
    }
    hasSampled.toSeq
  }
}
