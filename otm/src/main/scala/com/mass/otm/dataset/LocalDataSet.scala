package com.mass.otm.dataset

import java.io.{BufferedReader, InputStreamReader}

import scala.collection.mutable
import scala.util.Using

import com.mass.otm.dataset.OTMSample.{OTMEvalSample, OTMTrainSample}
import com.mass.otm.{encoding, paddingIdx, upperLog2}
import com.mass.sparkdl.utils.{FileReader => DistFileReader}
import org.apache.commons.lang3.math.NumberUtils

class LocalDataSet(dataPath: String, seqLen: Int, minSeqLen: Int, splitRatio: Double) {
  import LocalDataSet._
  require(seqLen > 0 && minSeqLen > 0 && seqLen >= minSeqLen)

  private val initData: Seq[InitSample] = readFile(dataPath)
  lazy val uniqueItems: Seq[Int] = initData.map(_.item).distinct
  lazy val (itemIdMapping, idItemMapping) = initializeMapping(uniqueItems)
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

  def numItem: Int = uniqueItems.length

  def numNode: Int = {
    val leafLevel = upperLog2(numItem)
    (math.pow(2, leafLevel + 1) - 1).toInt
  }
}

object LocalDataSet {

  case class InitSample(user: Int, item: Int, timestamp: Long)

  case class DataInfo(
    userConsumed: Map[Int, Seq[Int]],
    trainData: IndexedSeq[OTMSample],
    evalData: IndexedSeq[OTMSample]
  )

  def apply(dataPath: String, seqLen: Int, minSeqLen: Int, splitRatio: Double): LocalDataSet = {
    new LocalDataSet(dataPath, seqLen, minSeqLen, splitRatio)
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
      .map(i => InitSample(i(0).toInt, i(1).toInt, i(3).toLong))
      .toSeq
  }

  def initializeMapping(items: Seq[Int]): (Map[Int, Int], Map[Int, Int]) = {
    val shuffledItems = scala.util.Random.shuffle(items)
    val sampledIds = sampleRandomLeaves(items.length)
    (shuffledItems.zip(sampledIds).toMap, sampledIds.zip(shuffledItems).toMap)
  }

  private def sampleRandomLeaves(itemNum: Int): Seq[Int] = {
    val leafLevel = upperLog2(itemNum)
    val start = math.pow(2, leafLevel).toInt - 1
    val end = start * 2 + 1
    val hasSampled = mutable.BitSet.empty
    while (hasSampled.size < itemNum) {
      val s = scala.util.Random.between(start, end)
      if (!hasSampled.contains(s)) {
        hasSampled += s
      }
    }
    hasSampled.toSeq
  }
}
