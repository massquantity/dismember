package com.mass.dr.dataset

import java.io.{BufferedInputStream, DataInputStream}
import java.util.concurrent.atomic.AtomicInteger

import scala.collection.mutable
import scala.util.{Failure, Random, Success, Using}

import com.mass.dr.{encoding, Path => DRPath}
import com.mass.dr.dataset.DRSample.{DREvalSample, DRTrainSample}
import com.mass.dr.protobuf.item_mapping.{ItemSet, Path => ProtoPath}
import com.mass.sparkdl.dataset.DataUtil
import com.mass.sparkdl.utils.{FileReader => DistFileReader}
import org.apache.commons.lang3.math.NumberUtils

class LocalDataSet(
    numLayer: Int,
    numNode: Int,
    numPathPerItem: Int,
    trainBatchSize: Int,
    evalBatchSize: Int,
    seqLen : Int,
    minSeqLen: Int,
    dataPath: String,
    mappingPath: Option[String],
    splitRatio: Double = 0.8,
    delimiter: String = ",",
    paddingIdx: Int = -1) {
  import LocalDataSet._
  require(seqLen > 0 && minSeqLen > 0 && seqLen >= minSeqLen)

  private val initData: Array[InitSample] = readFile(dataPath, delimiter)

  lazy val (itemIdMapping, itemPathMapping) = mappingPath match {
    case Some(path) => loadMapping(path)
    case None =>
      val uniqueItems = initData.map(_.item).distinct
      val idMapping = uniqueItems.zipWithIndex.toMap
      val pathMapping = initializeMapping(uniqueItems.length, numLayer, numNode, numPathPerItem)
      (idMapping, pathMapping)
  }

  lazy val idItemMapping: Map[Int, Int] = itemIdMapping.map(i => i._2 -> i._1)

  lazy val (userConsumed, trainData, evalData) = generateData()

  lazy val trainMiniBatch: MiniBatch = new MiniBatch(
    itemPathMapping,
    numItem,
    numNode,
    numLayer,
    numPathPerItem,
    seqLen
  )
  lazy val evalMiniBatch: MiniBatch = new MiniBatch(
    itemPathMapping,
    numItem,
    numNode,
    numLayer,
    numPathPerItem,
    seqLen
  )

  private[dr] def iteratorMiniBatch(train: Boolean): Iterator[MiniBatch] = {
    new Iterator[MiniBatch] {
      private val (miniBatch, batchSize, originalDataSize) =
        if (train) {
          (trainMiniBatch, trainBatchSize, trainData.length)
        } else {
          (evalMiniBatch, evalBatchSize, evalData.length)
        }
      private val numTargetsPerBatch = math.max(1, batchSize / numPathPerItem)
      private val index = new AtomicInteger(0)

      override def hasNext: Boolean = {
        if (train) true else index.get() < originalDataSize
      }

      override def next(): MiniBatch = {
        val curIndex = index.getAndAdd(numTargetsPerBatch)
        val offset = curIndex % originalDataSize
        val length = math.min(numTargetsPerBatch, originalDataSize - offset)
        miniBatch.updatePosition(offset, length)
      }
    }
  }

  private def generateData(): (Map[Int, Array[Int]], Array[DRSample], Array[DRSample]) = {
    val groupedSamples = initData.groupBy(_.user).toArray.map { case (user, samples) =>
      (user, samples.sortBy(_.timestamp).map(_.item).distinct.map(itemIdMapping(_)))
    }
    val (userConsumed, trainSamples, evalSamples) = groupedSamples.map { case (user, items) =>
      if (items.length <= minSeqLen) {
        (user -> items, Array.empty[DRTrainSample], Array.empty[DREvalSample])
      } else if (items.length == minSeqLen + 1) {
        (user -> items, Array(DRTrainSample(items.init, items.last)), Array.empty[DREvalSample])
      } else {
        val fullSeq = Array.fill[Int](seqLen - minSeqLen)(paddingIdx) ++ items
        val splitPoint = math.ceil((items.length - minSeqLen) * splitRatio).toInt
        val _trainSamples = fullSeq
          .slice(0, splitPoint + seqLen)
          .sliding(seqLen + 1)
          .map(s => DRTrainSample(s.init, s.last))
          .toArray

        val consumed = items.slice(0, splitPoint + minSeqLen)
        val consumedSet = consumed.toSet
        val (_evalSeq, _labels) = fullSeq.splitAt(splitPoint + seqLen)
        val labels = _labels.filterNot(consumedSet)  // remove items appeared in train data
        val _evalSamples =
          if (labels.nonEmpty) {
            val evalSeq = _evalSeq.takeRight(seqLen)
            Array(DREvalSample(evalSeq, labels, user))
          } else {
            Array.empty[DREvalSample]
          }
        (user -> consumed, _trainSamples, _evalSamples)
      }
    }.unzip3

    (userConsumed.toMap,
     trainSamples.filter(_.nonEmpty).flatten,
     evalSamples.filter(_.nonEmpty).flatten)
  }

  def shuffle(): Unit = DataUtil.shuffle(trainData)

  def trainSize: Int = trainData.length

  def getTrainData: Array[DRSample] = trainData

  def getEvalData: Array[DRSample] = evalData

  def getUserConsumed: Map[Int, Array[Int]] = userConsumed

  def numItem: Int = itemIdMapping.size

}

object LocalDataSet {

  case class InitSample(user: Int, item: Int, timestamp: Long)

  def buildTrainData(
      dataPath: String,
      seqLen : Int,
      minSeqLen: Int,
      delimiter: String,
      paddingIdx: Int): Array[DRSample] = {
    val groupedSamples = readFile(dataPath, delimiter).groupBy(_.user)
    groupedSamples
      .map(s => s._2.sortBy(_.timestamp).map(_.item).distinct)
      .withFilter(_.length > minSeqLen)
      .flatMap(ss => {
        val items = if (seqLen > minSeqLen) Array.fill[Int](seqLen - minSeqLen)(paddingIdx) ++ ss else ss
        items.sliding(seqLen + 1).map(sss => DRTrainSample(sequence = sss.init, target = sss.last))
      }).toArray
  }

  private def readFile(dataPath: String, delimiter: String): Array[InitSample] = {
    val fileReader = DistFileReader(dataPath)
    val input = fileReader.open()
    val fileInput = scala.io.Source.fromInputStream(input, encoding.name)
    val samples = {
      for {
        line <- fileInput.getLines
        arr = line.trim.split(delimiter)
        if arr.length == 5 && NumberUtils.isCreatable(arr(0))
      } yield InitSample(arr(0).toInt, arr(1).toInt, arr(3).toLong)
    }.toArray
    fileInput.close()
    input.close()
    fileReader.close()
    samples
  }

  private def initializeMapping(
      numItem: Int,
      numLayer: Int,
      numNode: Int,
      numPathPerItem: Int): Map[Int, IndexedSeq[DRPath]] = {
    (0 until numItem).map(_ -> {
      val m = for {
        _ <- 1 to numPathPerItem
        _ <- 1 to numLayer
      } yield Random.nextInt(numNode)
      m.sliding(numLayer, numLayer).toIndexedSeq
    }).toMap
  }

  private[dr] def loadMapping(path: String): (Map[Int, Int], Map[Int, Seq[DRPath]]) = {
    val idMapping = mutable.Map.empty[Int, Int]
    val pathMapping = mutable.Map.empty[Int, Seq[DRPath]]
    val fileReader = DistFileReader(path)
    val input = fileReader.open()
    Using(new DataInputStream(new BufferedInputStream(input))) { input =>
      val size = input.readInt()
      val buf = new Array[Byte](size)
      input.readFully(buf)
      val allItems = ItemSet.parseFrom(buf).items
      allItems.foreach { i =>
        idMapping(i.item) = i.id
        pathMapping(i.id) = i.paths.map(_.index.toIndexedSeq)
      }
    } match {
      case Success(_) =>
        input.close()
        fileReader.close()
      case Failure(t: Throwable) =>
        throw t
    }
    (Map.empty ++ idMapping, Map.empty ++ pathMapping)
  }
}
