package com.mass.dr.dataset

import java.util.concurrent.atomic.AtomicInteger

import com.mass.dr.{encoding, Path}
import com.mass.sparkdl.dataset.DataUtil
import com.mass.sparkdl.utils.{FileReader => DistFileReader}
import org.apache.commons.lang3.math.NumberUtils

class LocalDataSet(
    itemPathMapping: Map[Int, Seq[Path]],
    numLayer: Int,
    numPathPerItem: Int,
    batchSize: Int,
    seqLen : Int,
    minSeqLen: Int,
    numThreads: Int,
    dataPath: String,
    delimiter: String = ",") {
  import LocalDataSet._
  require(seqLen > 0 && minSeqLen > 0 && seqLen >= minSeqLen)

  private val dataBuffer: Array[DRSample] = generateData(
    dataPath,
    seqLen,
    minSeqLen,
    delimiter
  )
  private val miniBatch: MiniBatch = new MiniBatch(
    itemPathMapping,
    numLayer,
    numPathPerItem,
    batchSize,
    seqLen,
    dataBuffer.length
  )

  def shuffle(): Unit = {
    DataUtil.shuffle(dataBuffer)
  }

  def size(): Int = dataBuffer.length

  def iteratorMiniBatch(): Iterator[MiniBatch] = {
    new Iterator[MiniBatch] {
      private val _miniBatch = miniBatch
      private val numTargetsPerBatch = _miniBatch.numTargetsPerBatch
      private val index = new AtomicInteger(0)

      override def hasNext: Boolean = true

      override def next(): MiniBatch = {
        val curIndex = index.getAndAdd(numTargetsPerBatch)
        val offset = curIndex % _miniBatch.originalDataSize
        val length = math.min(numTargetsPerBatch, _miniBatch.originalDataSize - offset)
        _miniBatch.updatePosition(offset, length)
      }
    }
  }

  def getData: Array[DRSample] = dataBuffer
}

object LocalDataSet {

  case class InitSample(user: Int, item: Int, timestamp: Long)

  def generateData(
      dataPath: String,
      seqLen : Int,
      minSeqLen: Int,
      delimiter: String): Array[DRSample] = {
    val groupedSamples = readFile(dataPath, delimiter).toArray.groupBy(_.user)
    groupedSamples
      .map(s => s._2.sortBy(_.timestamp).map(_.item).distinct)
      .withFilter(_.length > minSeqLen)
      .flatMap(ss => {
        val items = if (seqLen > minSeqLen) Array.fill[Int](seqLen - minSeqLen)(0) ++ ss else ss
        items.sliding(seqLen + 1).map(sss => DRSample(sequence = sss.init, target = sss.last))
      }).toArray
  }

  def readFile(dataPath: String, delimiter: String): Iterator[InitSample] = {
    val fileReader = DistFileReader(dataPath)
    val input = fileReader.open()
    val fileInput = scala.io.Source.fromInputStream(input, encoding.name)
    val samples =
      for {
        line <- fileInput.getLines
        arr = line.trim.split(delimiter)
        if arr.length == 5 && NumberUtils.isCreatable(arr(0))
      } yield InitSample(arr(0).toInt, arr(1).toInt, arr(3).toLong)

    fileInput.close()
    input.close()
    fileReader.close()
    samples
  }
}
