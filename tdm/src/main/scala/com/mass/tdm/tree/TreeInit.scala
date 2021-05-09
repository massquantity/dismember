package com.mass.tdm.tree

import java.io._
import java.nio.file.{Files, Paths}

import scala.collection.mutable
import scala.io.Source

import com.mass.tdm.utils.Context
import org.apache.spark.sql.functions.udf

// https://github.com/databricks/scala-style-guide/blob/master/README-ZH.md
class TreeInit(seqLen: Int, minSeqLen: Int, splitForEval: Boolean, splitRatio: Double = 0.8) {
  import TreeInit.{InitSample, Item}
  require(seqLen > 0 && minSeqLen > 0 && seqLen >= minSeqLen)

  def generate(
      dataFile: String,
      trainFile: String,
      evalFile: Option[String],
      statFile: String,
      leafIdFile: String,
      treePbFile: String): Unit = {

    if (splitForEval) {
      require(evalFile.isDefined, "couldn't find eval file path...")
    }
    require(Files.exists(Paths.get(dataFile)), s"$dataFile doesn't exist")

    val trainSample = readFile(dataFile)
    val userInteraction = userInteracted(trainSample)
    val stat = writeFile(userInteraction, trainFile, evalFile, statFile)
    initializeTree(trainSample, stat, leafIdFile, treePbFile)
  }

  def readFile(filename: String): InitSample = {
    var categoryDict = Map.empty[String, Int]
    var labelDict = Map.empty[String, Float]
    val users = new mutable.ArrayBuffer[Int]
    val items = new mutable.ArrayBuffer[Int]
    val categories = new mutable.ArrayBuffer[Int]
    val labels = new mutable.ArrayBuffer[Float]
    val timestamp = new mutable.ArrayBuffer[Long]

    val fileInput = Source.fromFile(filename)
    for {
      line <- fileInput.getLines
      arr = line.trim.split(",")
      if arr.length == 5 && arr(0) != "user"
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

    InitSample(
      users.toArray,
      items.toArray,
      categories.toArray,
      labels.toArray,
      timestamp.toArray)
  }

  private def userInteracted(trainSample: InitSample): Map[Int, Array[Int]] = {
    val interactions = mutable.Map.empty[Int, List[(Int, Long)]]
    // (user, item, time).zipped
    trainSample.user.zip(trainSample.item).zip(trainSample.timestamp) foreach {
      case ((u: Int, i: Int), t: Long) =>
        interactions(u) = (i, t) :: interactions.getOrElse(u, Nil)
      case _ =>
        throw new IllegalArgumentException
    }
    val res = interactions.map { case (user, items) =>
      val sortedItems = items.sortBy(_._2).map(_._1).toArray
      (user, sortedItems)
    }
    Map.empty ++ res
  }

  private def writeFile(
      userInteraction: Map[Int, Array[Int]],
      trainFile: String,
      evalFile: Option[String],
      statFile: String): Map[Int, Int] = {

    val stat = mutable.Map.empty[Int, Int]
    val writer = new BufferedWriter(new PrintWriter(new File(trainFile)))
    val evalWriter = {
      if (splitForEval) {
        new BufferedWriter(new PrintWriter(new File(evalFile.get)))
      } else {
        null
      }
    }

    try {
      if (!splitForEval) {
        userInteraction.foreach {
          case (user, items)  =>
            if (items.length >= minSeqLen) {
              val arr = Array.fill[Int](seqLen - minSeqLen)(0) ++ items
              var ui = 0
              arr.sliding(seqLen) foreach { seq =>
                writer.write(s"${user}_$ui")
                seq.foreach(i => writer.write(s",$i")) // if (i != 0)
                writer.write("\n")
                val targetItem = seq.last
                stat(targetItem) = stat.getOrElse(targetItem, 0) + 1
                ui += 1
              }
            }
        }
      } else {
        val users = userInteraction.keys.toArray
        var i = 0
        while (i < users.length) {
          val user = users(i)
          val items = userInteraction(user)
          if (items.length >= (minSeqLen + 1)) {
            val arr = Array.fill[Int](seqLen - minSeqLen)(0) ++ items
            val splitPoint = math.max(1, (items.length * splitRatio - minSeqLen + 1).toInt)
            var i = 0
            while (i < splitPoint) {
              writer.write(s"${user}_$i")
              var s = i
              val end = i + seqLen
              while (s < end) {
                writer.write(s",${arr(s)}")
                s += 1
              }
              writer.write("\n")
              val targetItem = arr(i + seqLen - 1)
              stat(targetItem) = stat.getOrElse(targetItem, 0) + 1
              i += 1
            }

            evalWriter.write(s"${user}")
            while (i < arr.length) {
              evalWriter.write(s",${arr(i)}")
              i += 1
            }
            evalWriter.write("\n")
          }
          i += 1
        }
      }
    } catch {
      case e: IOException =>
        throw e
      case t: Throwable =>
        throw t
    } finally {
      writer.close()
      if (null != evalWriter){
        evalWriter.close()
      }
    }

    val writerStat = new PrintWriter(new File(statFile))
    try {
      stat.foreach {
        case (user, count) =>
          writerStat.write(s"$user, $count\n")
        case _ =>
      }
    } catch {
      case e: IOException =>
        throw e
      case t: Throwable =>
        throw t
    } finally {
      writerStat.close()
    }

    Map.empty ++ stat
  }

  private[tdm] def initializeTree(
      trainSample: InitSample,
      stat: Map[Int, Int],
      leafIdFile: String,
      treePbFile: String): Unit = {

    // val item_id_set = trainSample("item").distinct.toSet[Int]
    val itemSet = mutable.HashSet.empty[Int]
    var items = mutable.ArrayBuffer.empty[Item]
    trainSample.item zip trainSample.category foreach {
      case (item_id: Int, cat_id: Int) =>
        if (!itemSet.contains(item_id)) {
          itemSet += item_id
          items += Item(item_id, cat_id)
        }
      case _ =>
    }

    val writer = new BufferedWriter(new FileWriter(new File(leafIdFile)))
    try itemSet.foreach(x => writer.write(s"${x.toString}\n"))
    finally writer.close()

    items = items.sortWith((a, b) => {
      a.catId < b.catId || (a.catId == b.catId && a.itemId < b.itemId)
    })

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

    def genCode2(start: Int, end: Int, code: Int): Unit = {
      if (end == start + 1) {
        items(start).code = code
      } else if (end > start + 1) {
        val mid = (start + end) >>> 1
        genCode2(mid, end, 2 * code + 1)
        genCode2(start, mid, 2 * code + 2)
      }
    }

    genCode(0, items.length, 0)
    val ids = items.map(_.itemId).toArray
    val codes = items.map(_.code).toArray

    val builder = new TreeBuilder(treePbFile)
    builder.build(
      treeIds = ids,
      treeCodes = codes,
      stat = Some(stat)
    )
  }
}

object TreeInit extends Context {

  case class InitSample(
    user: Array[Int],
    item: Array[Int],
    category: Array[Int],
    label: Array[Float],
    timestamp: Array[Long])

  case class Item(itemId: Int, catId: Int, var code: Int = 0)

  def readFileSpark(filename: String): Map[String, List[Any]] = {
    import spark.implicits._
    def isHeader(line: String): Boolean = line.contains("user")
    val file = spark.read.textFile(filename)
      .filter(!isHeader(_))
      .selectExpr("split(trim(value), ',') as col")
      .selectExpr(
        "cast(col[0] as int) as user_id",
        "cast(col[1] as int) as itemId",
        "cast(col[2] as int) as rating",
        "cast(col[3] as long) as timestamp",
        "cast(col[4] as string) as genre"
      ).limit(7)

    val behaviorMap = (1 to 5).zip(0 to 4).toMap
    def mapBehavior = udf((rating: Int) => behaviorMap(rating))

    val categoryMap = file.map(_.getString(4)).rdd.distinct().zipWithUniqueId().collectAsMap()
    def mapCategory = udf((category: String) => categoryMap(category))
    // println(s"type of categoryMap is ${getType(categoryMap)} and ${categoryMap.getClass}")

    var train_sample = Map.empty[String, List[Any]]
    train_sample += ("USER_ID" -> file.map(_.getAs[Int]("user_id")).collect().toList)
    train_sample += ("ITEM_ID" -> file.map(_.getAs[Int]("itemId")).collect().toList)
    train_sample += ("BEHAVIOR" ->
      file.withColumn("behavior", mapBehavior($"rating"))
        .select("behavior")
        .map(_.getAs[Int]("behavior"))
        .collect()
        .toList
      )
    train_sample += ("CAT_ID" ->
      file.withColumn("category", mapCategory($"genre"))
        .select("category")
        .map(_.getLong(0).toInt)
        .collect()
        .toList
      )
    train_sample += ("TIMESTAMP" -> file.map(_.getAs[Long]("timestamp")).collect().toList)
    train_sample
  }

}
