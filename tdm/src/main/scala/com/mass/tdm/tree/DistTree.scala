package com.mass.tdm.tree

import java.io.{BufferedInputStream, DataInputStream, EOFException, FileNotFoundException}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Using

import com.mass.sparkdl.utils.{FileReader => DistFileReader}
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

  @inline
  def isFiltered(code: Int): Boolean = {
    if (code < 0) return true
    var maxIdx = 0
    var i = 0
    while (i < maxLevel) {
      maxIdx = maxIdx * 2 + 1
      i += 1
    }

    var _code = code
    while (_code < maxIdx) {
      if (leafCodes.contains(_code)) {
        return false
      }
      _code = _code * 2 + 1
    }
    true
  }

  def loadItems(idCodeAllParts: ArrayBuffer[IdCodePart], meta: TreeMeta): Unit = {
    var _maxCode: Int = -1
    var maxLeafId: Int = -1

    idCodeAllParts.foreach { part =>
      part.idCodeList.foreach { i =>
        idCodeMap(i.id) = i.code
        leafCodes += i.code
        maxLeafId = math.max(maxLeafId, i.id)
        _maxCode = math.max(_maxCode, i.code)
      }
    }

    nonLeafOffset = maxLeafId + 1
    maxCode = _maxCode
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
