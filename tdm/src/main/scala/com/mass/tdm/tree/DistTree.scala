package com.mass.tdm.tree

import java.io.{BufferedInputStream, DataInputStream, EOFException, FileNotFoundException}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Using

import com.mass.scalann.utils.{FileReader => DistFileReader}
import com.mass.tdm.encoding
import com.mass.tdm.protobuf.store_kv.KVItem
import com.mass.tdm.protobuf.tree.{IdCodePart, Node, TreeMeta}

trait DistTree {
  import com.mass.tdm.tree.DistTree.multiPut

  protected val codeNodeMap: mutable.Map[Int, Node]
  protected val idCodeMap: mutable.Map[Int, Int]
  protected val leafCodes: mutable.BitSet
  protected var initialized: Boolean
  protected var maxLevel: Int
  protected var maxCode: Int
  protected var nonLeafOffset: Int

  def loadItems(idCodeAllParts: ArrayBuffer[IdCodePart], meta: TreeMeta): Unit = {
    val pairs =
      for {
        part <- idCodeAllParts
        pair <- part.idCodeList
      } yield pair

    val (leafIds, codes) = (pairs.map(_.id), pairs.map(_.code))
    idCodeMap ++= leafIds.zip(codes)
    leafCodes ++= codes
    nonLeafOffset = leafIds.max + 1
    maxCode = codes.max
    maxLevel = meta.maxLevel
  }

  def loadData(path: String): (ArrayBuffer[IdCodePart], TreeMeta) = {
    val kBatchSize = 500
    val idCodeAllParts = ArrayBuffer.empty[IdCodePart]
    var treeMeta: TreeMeta = null
    val keys = new ArrayBuffer[Int]()
    val values = new ArrayBuffer[Node]()
    val fileReader = DistFileReader(path)
    val input = fileReader.open()

    Using(new DataInputStream(new BufferedInputStream(input))) { input =>
      while (input.available() > 0) {
        val num = input.readInt()
        val buf = new Array[Byte](num)
        // input.read(buf, 0, num)
        input.readFully(buf)
        val item = KVItem.parseFrom(buf)
        val key = item.key.toString(encoding.name())
        val value = item.value.toByteArray
        if (key.startsWith("tree_meta")) {
          treeMeta = TreeMeta.parseFrom(value)
        } else if (key.startsWith("Part_")) {
          idCodeAllParts += IdCodePart.parseFrom(value)
        } else {
          keys += item.key.toString(encoding.name()).toInt
          values += Node.parseFrom(item.value.toByteArray)
          if (keys.length >= kBatchSize) {
            multiPut(codeNodeMap, keys, values)
            keys.clear()
            values.clear()
          }
        }
      }
    }.recover {
      case e: FileNotFoundException =>
        println(s"file $path not found")
        throw e
      case _: EOFException =>
        println(s"file: $path read ended")
      case t: Throwable =>
        throw t
    }.get

    if (keys.nonEmpty) {
      multiPut(codeNodeMap, keys, values)
    }

    (idCodeAllParts, treeMeta)
  }
}

object DistTree {

  case class TreeNode(code: Int, node: Node)

  private def multiPut(
      map: mutable.Map[Int, Node],
      keys: ArrayBuffer[Int],
      values: ArrayBuffer[Node]): Unit = {

    keys.zip(values) foreach {
      case (k: Int, v: Node) => map(k) = v
    }
  }
}
