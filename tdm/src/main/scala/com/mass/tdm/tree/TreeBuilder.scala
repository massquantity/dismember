package com.mass.tdm.tree

import java.io._
import java.nio.ByteBuffer

import scala.annotation.tailrec
import scala.collection.mutable
import scala.util.{Failure, Success, Using}

import com.google.protobuf.ByteString
import com.mass.sparkdl.utils.{FileWriter => DistFileWriter}
import com.mass.tdm.encoding
import com.mass.tdm.protobuf.store_kv.KVItem
import com.mass.tdm.protobuf.tree.{IdCodePair, IdCodePart, Node, TreeMeta}
import org.apache.log4j.Logger

object TreeBuilder {

  val logger: Logger = Logger.getLogger(getClass)

  case class Item(id: Int, code: Int)

  def build(
    outputTreePath: String,
    treeIds: Array[Int],
    treeCodes: Array[Int],
    stat: Option[Map[Int, Int]] = None
  ): Unit = {
    val offset = math.max(0, treeIds.max) + 1
    val log2 = (n: Int) => math.floor(math.log(n) / math.log(2)).toInt
    val maxLevel = log2(treeCodes.max + 1)
    val minLeafCode = math.pow(2, maxLevel).toInt - 1
    val leafCodes = flattenLeaves(treeCodes, minLeafCode)
    val items = treeIds.lazyZip(leafCodes).map(Item).sortBy(_.code)

    val pstat = stat match {
      case Some(treeStat) => computeNodeOccurrence(items, treeStat, maxLevel)
      case None => Map.empty[Int, Float]
    }

    val fileWriter = DistFileWriter(outputTreePath)
    val output: OutputStream = fileWriter.create(overwrite = true)
    Using(new BufferedOutputStream(output)) { writer =>
      val parts = new mutable.ArrayBuffer[IdCodePart]()
      val tmpItems = new mutable.ArrayBuffer[IdCodePair]()
      val savedItems = new mutable.HashSet[Int]()

      items.zipWithIndex foreach { case (Item(id, code), i) =>
        val prob = stat match {
          case Some(s) if s.contains(id) => s(id).toFloat
          case _ => 1.0f
        }
        val leafCatId = 0
        val isLeaf = true
        val leafNode = Node(id, prob, leafCatId, isLeaf)
        val leafKV = KVItem(toByteString(code), leafNode.toByteString)
        writeKV(leafKV, writer)

        tmpItems += IdCodePair(id, code)
        if (i == items.length - 1 || tmpItems.length == 512) {
          val partId = "Part_" + (parts.length + 1)
          parts += IdCodePart(toByteString(partId), tmpItems.clone().toSeq)
          tmpItems.clear()
        }

        val ancestors = getAncestors(code, maxLevel)
        ancestors foreach { ancCode =>
          if (!savedItems.contains(ancCode)) {
            val id = ancCode + offset
            val prob = pstat.getOrElse(ancCode, 1.0f)
            val leafCatId = 0
            val isLeaf = false
            val node = Node(id, prob, leafCatId, isLeaf)
            val ancestorKV = KVItem(toByteString(ancCode), node.toByteString)
            writeKV(ancestorKV, writer)
            savedItems += ancCode
          }
        }
      }

      parts.foreach { p =>
        val partKV = KVItem(p.partId, p.toByteString)
        writeKV(partKV, writer)
      }

      val partIds = parts.map(x => x.partId).toArray
      val meta = TreeMeta(maxLevel, partIds)
      val metaKV = KVItem(toByteString("tree_meta"), meta.toByteString)
      writeKV(metaKV, writer)
      logger.info(s"item num: ${treeIds.length}, tree level: $maxLevel, " +
        s"leaf code start: ${leafCodes.min}, leaf code end: ${leafCodes.max}")

    } match {
      case Success(_) =>
        output.close()
        fileWriter.close()
      case Failure(e: FileNotFoundException) =>
        println(s"""file "$outputTreePath" not found""")
        throw e
      case Failure(t: Throwable) =>
        throw t
    }
  }

  def getAllocatedLong(i: Long): Long = {
    val bf: ByteBuffer = allocateByteLong(i)
    ByteBuffer.wrap(bf.array()).getLong()
  }

  def allocateByteLong(i: Long): ByteBuffer = {
    ByteBuffer.allocate(8).putLong(i)
  }

  def allocateByteInt(i: Int): ByteBuffer = {
    ByteBuffer.allocate(4).putInt(i)
  }

  def toByteString[T](key: T): ByteString = {
    key match {
      case i: String => ByteString.copyFrom(i, encoding.name())
      // case i: Long => ByteString.copyFrom(allocateByteLong(i).array())
      // case i: Int => ByteString.copyFrom(allocateByteInt(i).array())
      case i => ByteString.copyFrom(i.toString, encoding.name())
    }
  }

  def writeKV(message: KVItem, writer: OutputStream): Unit = {
    // writer.writeInt(message.serializedSize)
    val bf: ByteBuffer = allocateByteInt(message.serializedSize)
    writer.write(bf.array())
    message.writeTo(writer)
  }

  // make all leaf nodes at same level
  def flattenLeaves(codes: Array[Int], minCode: Int): Array[Int] = {
    @tailrec
    def sink(code: Int): Int = {
      if (code >= minCode) code
      else sink(code * 2 + 1)
    }
    codes.map(sink)
  }

  def getAncestors(code: Int, maxLevel: Int): List[Int] = {
    val ancestors = List.fill(maxLevel)(0)
    ancestors.scanLeft(code)((a, _) => (a - 1) / 2).tail
  }

  def computeNodeOccurrence(items: Array[Item], stat: Map[Int, Int], maxLevel: Int): Map[Int, Float] = {
    val res = mutable.Map.empty[Int, Float]
    items foreach { case Item(id, code) =>
      val ancestors = getAncestors(code, maxLevel)
      ancestors foreach { anc =>
        if (stat.contains(id)) {
          res(anc) = res.getOrElse(anc, 0.0f) + stat(id)
        }
      }
    }
    res.toMap
  }
}
