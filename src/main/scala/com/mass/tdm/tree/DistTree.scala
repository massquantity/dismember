package com.mass.tdm.tree

import java.io._
import java.nio.channels.FileChannel
import java.nio.charset.Charset
import java.nio.file.Paths

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Using

import com.mass.tdm.encoding
import com.mass.tdm.protobuf.store_kv.KVItem
import com.mass.tdm.protobuf.tree.{IdCodePair, IdCodePart, TreeMeta}

class DistTree extends Serializable {
  import DistTree.{TreeNode, loadData, loadItems}

  val kvData = mutable.HashMap.empty[String, Array[Byte]]
  val idCodeMap = mutable.HashMap.empty[Long, Long]
  val allCodes = mutable.HashSet.empty[Long]
  var initialized: Boolean = false
  var maxLevel: Int = 0
  var maxCode: Long = -1
  var nonLeafOffset: Long = -1

  def multiGet(keys: Array[String]): Array[Array[Byte]] = {
    keys.map(kvData(_))
  }

  def multiPut(keys: ArrayBuffer[String], values: ArrayBuffer[Array[Byte]]): Unit = {
    keys.zip(values) foreach {
      case (k: String, v: Array[Byte]) => kvData(k) = v
    }
  }

  def init(path: String): Unit = {
    loadData(path, this)
    loadItems(this)
  }

  @inline
  def isFiltered(code: Long): Boolean = {
    if (code < 0) return true
    var maxIdx = 0
    var i = 0
    while (i < maxLevel) {
      maxIdx = maxIdx * 2 + 1
      i += 1
    }

    var _code = code
    while (_code < maxIdx) {
      if (allCodes.contains(_code)) {
        return false
      }
      _code = _code * 2 + 1
    }
    true
  }

  def idToCode(itemIds: Array[Long]): Array[Long] = {
    require(initialized, "tree hasn't been initialized...")
    val res = Array.fill[Long](itemIds.length)(-1)
    var i = 0
    while (i < itemIds.length) {
      val id = itemIds(i)
      if (id == 0) {
        res(i) = 0
      } else if (id < nonLeafOffset && idCodeMap.contains(id)) {
        res(i) = idCodeMap(id)
      } else {
        res(i) = id - nonLeafOffset
        if (res(i) > maxCode) res(i) = -1
      }
      i += 1
    }
    res
  }

  def ancestorCodes(code: Long): Array[Long] = {
    if (code == 0 || isFiltered(code))
      return Array.emptyLongArray

    val res = new ArrayBuffer[Long]()
    var _code = code
    while (_code != 0) {
      res += _code
      _code = (_code - 1) >>> 1
    }
    res.toArray
  }

  def getAncestorNodes(itemCodes: Array[Long]): Array[Array[TreeNode]] = {
    require(initialized, "tree hasn't been initialized...")
    val res = new Array[Array[TreeNode]](itemCodes.length)
    itemCodes.zipWithIndex.foreach { case (code, i) =>
      val ancCodes = ancestorCodes(code)
      res(i) = ancCodes.map(ac => TreeNode(ac, kvData(ac.toString)))
    }
    res
  }
}

object DistTree {

  case class TreeNode(code: Long, node: Array[Byte])

  def apply(pbFilePath: String): DistTree = {
    val tree = new DistTree
    tree.init(pbFilePath)
    tree
  }

  def loadItems(tree: DistTree): Unit = {
    val meta: TreeMeta = TreeMeta.parseFrom(tree.kvData("tree_meta"))
    tree.maxLevel = meta.maxLevel
    var maxLeafId: Long = -1
    meta.idCodePart.foreach { p =>
      val partId = p.toString(encoding)
      val part = IdCodePart.parseFrom(tree.kvData(partId))
      part.idCodeList.foreach { i =>
        tree.idCodeMap(i.id) = i.code
        tree.allCodes += i.code
        maxLeafId = math.max(maxLeafId, i.id)
        tree.maxCode = math.max(tree.maxCode, i.code)
      }
    }
    tree.nonLeafOffset = maxLeafId + 1
    tree.initialized = true
    println(s"Load successfully, leaf node count: ${tree.idCodeMap.size}, " +
      s"nonLeafOffset: ${tree.nonLeafOffset}")
  }

  def loadData(path: String, tree: DistTree): Unit = {
    val kBatchSize = 500
    val keys = new ArrayBuffer[String]()
    val values = new ArrayBuffer[Array[Byte]]()
    Using(new DataInputStream(new BufferedInputStream(new FileInputStream(path)))) { input =>
      while (input.available() > 0) {
        val num = input.readInt()
        val buf = new Array[Byte](num)
        // input.read(buf, 0, num)
        input.readFully(buf)
        val item = KVItem.parseFrom(buf)
        keys += item.key.toString(encoding)
        values += item.value.toByteArray
        if (keys.length >= kBatchSize) {
          tree.multiPut(keys, values)
          keys.clear()
          values.clear()
        }
      }
    }.recover {
      case e: FileNotFoundException =>
        println(s"file $path not found")
        throw e
      case _: EOFException =>
        println(s"file: $path read ended")
      case e: Throwable =>
        throw e
    }.get

    if (keys.nonEmpty) {
      tree.multiPut(keys, values)
    }

   /*
   val channel = FileChannel.open(Paths.get(path))
   val length = channel.size()
   val buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, length)
   var p = 0
   while (buffer.remaining() > 0) {
     val num = buffer.getInt()
     val buf = new Array[Byte](num)
     // input.read(buf, 0, num)
     buffer.get(buf)
     val item = KVItem.parseFrom(buf)
     keys += item.key.toString(encoding)
     values += item.value.toByteArray
     if (keys.length >= kBatchSize) {
       multiPut(keys, values)
       keys.clear()
       values.clear()
     }
   }
   */
  }
}
