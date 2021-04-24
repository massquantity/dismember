package com.mass.tdm.tree

import java.io._

import scala.collection.mutable
import scala.io.Source
import scala.reflect.ClassTag

import com.mass.tdm.utils.Context
import org.apache.spark.sql.functions.udf

// https://github.com/databricks/scala-style-guide/blob/master/README-ZH.md
// use view
class TreeInit(seqLen: Int = 10, minSeqLen: Int = 5) extends Context {
  import TreeInit.{InitSample, Item}

  def this(seqLen: Int) = this(seqLen, 2)

  def generate(
      inputFile: String,
      outputFile: String,
      statFile: String,
      leafIdFile: String,
      treePbFile: String): Unit = {

    val trainSample = readFile(inputFile)
    val userInteraction = userInteracted(trainSample)
    val stat = writeFile(userInteraction, outputFile, statFile)
    initializeTree(trainSample, stat, leafIdFile, treePbFile)
  }

  def readFile(filename: String): InitSample = {
    var categoryDict = Map.empty[String, Int]
    var labelDict = Map.empty[String, Float]
    val users = new mutable.ArrayBuffer[Long]
    val items = new mutable.ArrayBuffer[Long]
    val categories = new mutable.ArrayBuffer[Int]
    val labels = new mutable.ArrayBuffer[Float]
    val timestamp = new mutable.ArrayBuffer[Long]

    val fileInput = Source.fromFile(filename)
    for {
      line <- fileInput.getLines
      arr = line.stripMargin.split(",")
      if arr.length == 5 && arr(0) != "user"
    } {
      users += arr(0).toLong
      items += arr(1).toLong
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

  private def userInteracted(trainSample: InitSample): Map[Long, Array[Long]] = {
    val interactions = mutable.Map.empty[Long, List[(Long, Long)]]
    // (user, item, time).zipped
    trainSample.user.zip(trainSample.item).zip(trainSample.timestamp) foreach {
      case ((u: Long, i: Long), t: Long) =>
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
      userInteraction: Map[Long, Array[Long]],
      outputFile: String,
      statFile: String): Map[Long, Int] = {

    val stat = mutable.Map.empty[Long, Int]
    val writer = new PrintWriter(new File(outputFile))
    try {
      userInteraction.foreach {
        case (user, items) if items.length >= minSeqLen =>
            val arr = Array.fill[Long](seqLen - minSeqLen)(0L) ++ items
            var ui = 0
            arr.sliding(seqLen) foreach { seq =>
              writer.write(s"${user}_$ui")
              seq.foreach(i => writer.write(s",$i")) // if (i != 0)
              writer.write("\n")
              val targetItem = seq.last
              stat(targetItem) = stat.getOrElse(targetItem, 0) + 1
              ui += 1
            }
            /*
            (0 to arr.length - seqLen) foreach { i =>
              writer.write(s"${user}_$i")
              var s = i
              val end = i + seqLen
              while (s < end) {
              //  if (arr(s) != 0)
                writer.write(s",${arr(s)}")
                s += 1
              }
              writer.write("\n")

              val targetItem = arr(i + seqLen - 1)
              stat(targetItem) = stat.getOrElse(targetItem, 0) + 1
            }
            */
        case _ =>
      }
    } catch {
      case e: IOException =>
        throw e
    } finally {
      writer.close()
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
    } finally {
      writerStat.close()
    }

    Map.empty ++ stat
  }

  private[tdm] def initializeTree(
      trainSample: InitSample,
      stat: Map[Long, Int],
      leafIdFile: String,
      treePbFile: String): Unit = {

    // val item_id_set11 = trainSample("User").distinct.toSet[Int]
    val itemSet = new mutable.HashSet[Long]()
    var items = mutable.ArrayBuffer.empty[Item]
    trainSample.item zip trainSample.category foreach {
      case (item_id: Long, cat_id: Int) =>
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
      a.cat_id < b.cat_id || (a.cat_id == b.cat_id && a.item_id < b.item_id)
    })

    def genCode(start: Int, end: Int, code: Int): Unit = {
      if (end <= start) return
      if (end == start + 1) {
        items(start).code = code
        return
      }
      val mid = (start + end) >>> 1
      genCode(mid,   end, 2 * code + 1)
      genCode(start, mid, 2 * code + 2)
    }

    def genCode2(start: Int, end: Int, code: Int): Unit = {
      if (end == start + 1) {
        items(start).code = code
      } else if (end > start + 1) {
        val mid = (start + end) >>> 1
        genCode2(mid,   end, 2 * code + 1)
        genCode2(start, mid, 2 * code + 2)
      }
    }

    genCode(0, items.length, 0)
    val ids = items.map(_.item_id).toArray
    val codes = items.map(_.code).toArray
    val builder = new TreeBuilder(treePbFile)
    builder.build(
      treeIds = ids,
      treeCodes = codes,
      stat = Some(stat)
    )
  }

  def readFileSpark(filename: String): Map[String, List[Any]] = {
    import spark.implicits._
    def isHeader(line: String): Boolean = line.contains("user")
    val file = spark.read.textFile(filename)
      .filter(!isHeader(_))
      .selectExpr("split(trim(value), ',') as col")
      .selectExpr(
        "cast(col[0] as int) as user_id",
        "cast(col[1] as int) as item_id",
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
    train_sample += ("ITEM_ID" -> file.map(_.getAs[Int]("item_id")).collect().toList)
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

object TreeInit {

  case class InitSample(
    user: Array[Long],
    item: Array[Long],
    category: Array[Int],
    label: Array[Float],
    timestamp: Array[Long])

  case class Item(item_id: Long, cat_id: Int, var code: Long = 0)

}
