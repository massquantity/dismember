package com.mass.tdm.tree

import java.io._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import com.mass.scalann.utils.{FileReader => DistFileReader, FileWriter => DistFileWriter}
import org.apache.commons.lang3.math.NumberUtils

// https://github.com/databricks/scala-style-guide/blob/master/README-ZH.md
class TreeInit(seqLen: Int, minSeqLen: Int, splitForEval: Boolean, splitRatio: Double) {
  import TreeInit._
  require(seqLen > 0 && minSeqLen > 0 && seqLen >= minSeqLen)
  require(splitRatio > 0 && splitRatio < 1)

  private val stat = mutable.HashMap.empty[Int, Int]
  private val userConsumed = mutable.HashMap.empty[Int, Array[Int]]

  def generate(
    dataFile: String,
    trainFile: String,
    evalFile: Option[String],
    statFile: String,
    leafIdFile: String,
    treePbFile: String,
    userConsumedFile: Option[String]
  ): (Array[Int], Array[Int]) = {
    if (splitForEval) {
      require(evalFile.isDefined, "couldn't find eval file path...")
    }

    val trainSample = readFile(dataFile)
    val userInteraction = getUserInteracted(trainSample)
    if (splitForEval) {
      writeFile(trainFile, mode = "half_train", userInteraction)
      writeFile(evalFile.get, mode = "eval", userInteraction)
    } else {
      writeFile(trainFile, mode = "train", userInteraction)
    }
    writeFile(statFile, mode = "stat", userInteraction)

    userConsumedFile match {
      case Some(path) => writeFile(path, mode = "user_consumed", userInteraction)
      case None =>
    }
    initializeTree(trainSample, leafIdFile, treePbFile)
  }

  def readFile(filename: String): InitSample = {
    var categoryDict = Map.empty[String, Int]
    var labelDict = Map.empty[String, Float]
    val users = new mutable.ArrayBuffer[Int]
    val items = new mutable.ArrayBuffer[Int]
    val categories = new mutable.ArrayBuffer[Int]
    val labels = new mutable.ArrayBuffer[Float]
    val timestamp = new mutable.ArrayBuffer[Long]

    val fileReader = DistFileReader(filename)
    val input = fileReader.open()
    val fileInput = Source.fromInputStream(input)
    for {
      line <- fileInput.getLines()
      arr = line.trim.split(",")
      if arr.length == 5 && NumberUtils.isCreatable(arr(0))
    } {
      users += arr(0).toInt
      items += arr(1).toInt
      timestamp += arr(3).toLong
      if (!labelDict.contains(arr(2))) {
        val size = labelDict.size.toFloat
        labelDict += (arr(2) -> size)
      }
      labels += labelDict(arr(2))

      if (!categoryDict.contains(arr(4))) {
        val size = categoryDict.size
        categoryDict += (arr(4) -> size)
      }
      categories += categoryDict(arr(4))
    }
    fileInput.close()
    input.close()
    fileReader.close()

    InitSample(
      users.toArray,
      items.toArray,
      categories.toArray,
      labels.toArray,
      timestamp.toArray
    )
  }

  private def getUserInteracted(trainSample: InitSample): Map[Int, Array[Int]] = {
    val interactions = mutable.Map.empty[Int, ArrayBuffer[(Int, Long)]]
    var i = 0
    val length = trainSample.user.length
    while (i < length) {
      val user = trainSample.user(i)
      val item = trainSample.item(i)
      val time = trainSample.timestamp(i)
      if (interactions.contains(user)) {
        interactions(user) += Tuple2(item, time)
      } else {
        interactions(user) = ArrayBuffer((item, time))
      }
      i += 1
    }

    val res = interactions.map { case (user, items) =>
      val sortedUniqueItems = items.toArray.sortBy(_._2).map(_._1).distinct
      (user, sortedUniqueItems)
    }

    Map.empty ++ res
  }

  private def writeFile(
    filePath: String,
    mode: String,
    userInteraction: Map[Int, Array[Int]]
  ): Unit = {
    val fileWriter: DistFileWriter = DistFileWriter(filePath)
    val output: OutputStream = fileWriter.create(overwrite = true)
    val writer = new PrintWriter(output, true)
    try {
      mode match {
        case "train" =>
          writeTrain(
            writer,
            userInteraction,
            userConsumed,
            seqLen,
            minSeqLen,
            stat
          )
        case "half_train" =>
          writeEither(
            writer,
            userInteraction,
            userConsumed,
            seqLen,
            minSeqLen,
            splitRatio,
            stat,
            train = true
          )
        case "eval" =>
          writeEither(
            writer,
            userInteraction,
            userConsumed,
            seqLen,
            minSeqLen,
            splitRatio,
            stat,
            train = false
          )
        case "stat" =>
          writeStat(writer, stat)
        case "user_consumed" =>
          writeUserConsumed(writer, userConsumed)
        case other =>
          throw new IllegalArgumentException(s"$other mode is not supported")
      }
    } catch {
      case e: IOException =>
        throw e
      case t: Throwable =>
        throw t
    } finally {
      writer.close()
      output.close()
      fileWriter.close()
    }
  }

  private[tdm] def initializeTree(
    trainSample: InitSample,
    leafIdFile: String,
    treePbFile: String
  ): (Array[Int], Array[Int]) = {
    val uniqueItems = (trainSample.item lazyZip trainSample.category)
      .map(Item(_, _))
      .distinctBy(_.itemId)

    val fileWriter = DistFileWriter(leafIdFile)
    val output = fileWriter.create(overwrite = true)
    val writer = new DataOutputStream(new BufferedOutputStream(output))
    try {
      uniqueItems.foreach(x => writer.writeBytes(s"${x.itemId.toString}\n"))
    } finally {
      writer.close()
      output.close()
      fileWriter.close()
    }

    val items = uniqueItems.sortWith { (a, b) =>
      a.catId < b.catId || (a.catId == b.catId && a.itemId < b.itemId)
    }

    def genCode(start: Int, end: Int, code: Int): Unit = {
      if (end <= start) return
      if (end == start + 1) {
        items(start).code = code
        return
      }
      val mid = (start + end) >>> 1
      genCode(mid, end, 2 * code + 1)
      genCode(start, mid, 2 * code + 2)
    }

    genCode(0, items.length, 0)
    val ids = items.map(_.itemId)
    val codes = items.map(_.code)

    TreeBuilder.build(
      outputTreePath = treePbFile,
      treeIds = ids,
      treeCodes = codes,
      stat = Some(stat.toMap)
    )
    (ids, codes)
  }
}

object TreeInit {

  case class Item(itemId: Int, catId: Int, var code: Int = 0)

  case class InitSample(
    user: Array[Int],
    item: Array[Int],
    category: Array[Int],
    label: Array[Float],
    timestamp: Array[Long]
  )

  // write seqLen + 1 items (sequence + target)
  private def writeTrain(
    writer: PrintWriter,
    userInteraction: Map[Int, Array[Int]],
    userConsumed: mutable.HashMap[Int, Array[Int]],
    seqLen: Int,
    minSeqLen: Int,
    stat: mutable.HashMap[Int, Int]
  ): Unit = {
    userInteraction.foreach {
      case (user, items) =>
        userConsumed(user) = items
        if (items.length > minSeqLen) {
          val arr = Array.fill[Int](seqLen - minSeqLen)(0) ++ items
          var ui = 0
          arr.sliding(seqLen + 1) foreach { seq =>
            writer.write(s"${user}_$ui,")
         //   seq.foreach(i => writer.write(s",$i"))
            writer.println(seq.mkString(","))  // if (i != 0)

            val targetItem = seq.last
            stat(targetItem) = stat.getOrElse(targetItem, 0) + 1
            ui += 1
          }
        }
    }
  }

  private def writeEither(
    writer: PrintWriter,
    userInteraction: Map[Int, Array[Int]],
    userConsumed: mutable.HashMap[Int, Array[Int]],
    seqLen: Int,
    minSeqLen: Int,
    splitRatio: Double,
    stat: mutable.HashMap[Int, Int],
    train: Boolean
  ): Unit = {
    val sb = new mutable.StringBuilder
    val users = userInteraction.keys.toArray
    users.foreach { user =>
      val items = userInteraction(user)
      if (train && items.length <= minSeqLen) {
        userConsumed(user) = items
      } else if (train && items.length > minSeqLen) {
        val arr = Array.fill[Int](seqLen - minSeqLen)(0) ++ items
        val trainNum = math.ceil((items.length - minSeqLen) * splitRatio).toInt
        if (items.length == minSeqLen + 1) {
          userConsumed(user) = items
        } else {
          userConsumed(user) = items.slice(0, trainNum + minSeqLen)
        }

        var i = 0
        while (i < trainNum) {
          writer.write(s"user_${user}_$i")
        //  sb ++= arr.slice(i, i + seqLen + 1).mkString(",")
          var s = i
          val end = i + seqLen + 1
          while (s < end) {
            writer.write(s",${arr(s)}")
            s += 1
          }
          writer.println()

          val targetItem = arr(i + seqLen)
          stat(targetItem) = stat.getOrElse(targetItem, 0) + 1
          i += 1
        }
      } else if (!train && items.length > minSeqLen + 1) {
        val arr = Array.fill[Int](seqLen - minSeqLen)(0) ++ items
        val splitPoint = math.ceil((items.length - minSeqLen) * splitRatio).toInt
        val consumed = userConsumed(user).toSet
        sb ++= s"user_$user"
        var hasNew = false
        var i = splitPoint
        val seqEnd = splitPoint + seqLen
        while (i < arr.length) {
          if (i < seqEnd) {
            sb ++= s",${arr(i)}"
          } else if (!consumed.contains(arr(i))) {  // remove items appeared in train data
            hasNew = true
            sb ++= s",${arr(i)}"
          }
          i += 1
        }

        if (hasNew) {
          writer.println(sb.toString())
        }
        sb.clear()
      }
    }
  }

  private def writeStat(
    writer: PrintWriter,
    stat: mutable.HashMap[Int, Int]
  ): Unit = {
    stat.foreach {
      case (user, count) =>
        writer.println(s"$user, $count")
      case _ =>
    }
  }

  private def writeUserConsumed(
    writer: PrintWriter,
    userConsumed: mutable.Map[Int, Array[Int]]
  ): Unit = {
    userConsumed.foreach {
      case (user, items) =>
        writer.write(s"user_$user")
        items.foreach(i => writer.write(s",$i"))
        writer.println()
      case _ =>
    }
  }
}
