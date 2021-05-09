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

  private[tdm] var kvData: mutable.HashMap[String, Array[Byte]] = _
  private var idCodeMap: mutable.HashMap[Int, Int] = _
  private var allCodes: mutable.BitSet = _   // mutable.HashSet
  private[tdm] var initialized: Boolean = false
  private[tdm] var maxLevel: Int = 0
  private[tdm] var maxCode: Int = -1
  private var nonLeafOffset: Int = -1

  def getMaxLevel: Int = maxLevel

  def getIds: Array[Int] = idCodeMap.keys.toArray.sorted

  def getIdCodeMap: mutable.HashMap[Int, Int] = idCodeMap

  def init(path: String): Unit = {
    loadData(path, this)
    loadItems(this)
    initialized = true
  }

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
      if (allCodes.contains(_code)) {
        return false
      }
      _code = _code * 2 + 1
    }
    true
  }

  def idToCode(itemIds: Array[Int]): Array[Int] = {
    require(initialized, "tree hasn't been initialized...")
    val res = Array.fill[Int](itemIds.length)(-1)
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

  def idToCode(itemId: Int): Int = {
    if (itemId == 0) {
      0
    } else if (itemId < nonLeafOffset && idCodeMap.contains(itemId)) {
      idCodeMap(itemId)
    } else {
      val res = itemId - nonLeafOffset
      if (res > maxCode) -1 else res
    }
  }

  /*
  def ancestorNodes(code: Int): Array[TreeNode] = {
    if (code == 0 || isFiltered(code))
      return Array.empty[TreeNode]

    val res = new ArrayBuffer[TreeNode]()
    var _code = code
    while (_code != 0) {
      res += TreeNode(_code, kvData(_code.toString))
      _code = (_code - 1) >> 1
    }
    res.toArray
  }
   */

  def getAncestorNodes(itemCodes: Array[Int]): Array[Array[TreeNode]] = {
  //  val res = new Array[Array[TreeNode]](itemCodes.length)
  //  itemCodes.zipWithIndex.foreach { case (code, i) =>
  //    val ancCodes = ancestorCodes(code)
  //    res(i) = ancCodes.map(ac => TreeNode(ac, kvData(ac.toString)))
  //  }

    itemCodes.map(code => {
      if (code == 0 || isFiltered(code)) {
        Array.empty[TreeNode]
      } else {
        val res = new ArrayBuffer[TreeNode]()
        var _code = code
        while (_code != 0) {
          res += TreeNode(_code, kvData(_code.toString))
          _code = (_code - 1) >> 1
        }
        res.toArray
      }
    })
  }

  def getChildNodes(itemCode: Int): ArrayBuffer[TreeNode] = {
    val res = ArrayBuffer.empty[TreeNode]
    val childLeft = 2 * itemCode + 1
    if (!isFiltered(childLeft)) {
      res += TreeNode(childLeft, kvData(childLeft.toString))
    }
    val childRight = 2 * itemCode + 2
    if (!isFiltered(childRight)) {
      res += TreeNode(childRight, kvData(childRight.toString))
    }
    res
  }
}

object DistTree {

  case class TreeNode(code: Int, node: Array[Byte])

  def apply(pbFilePath: String): DistTree = {
    val tree = new DistTree
    tree.init(pbFilePath)
    tree
  }

  private def loadItems(tree: DistTree): Unit = {
    val _kvData = tree.kvData
    val _idCodeMap = mutable.HashMap.empty[Int, Int]
    val _allCodes = mutable.BitSet.empty
    var _maxCode: Int = -1
    var maxLeafId: Int = -1
    val meta: TreeMeta = TreeMeta.parseFrom(_kvData("tree_meta"))

    meta.idCodePart.foreach { p =>
      val partId = p.toString(encoding.name())
      val part = IdCodePart.parseFrom(_kvData(partId))
      part.idCodeList.foreach { i =>
        _idCodeMap(i.id) = i.code
        _allCodes += i.code
        maxLeafId = math.max(maxLeafId, i.id)
        _maxCode = math.max(_maxCode, i.code)
      }
    }

    tree.nonLeafOffset = maxLeafId + 1
    tree.idCodeMap = _idCodeMap
    tree.allCodes = _allCodes
    tree.maxCode = _maxCode
    tree.maxLevel = meta.maxLevel
 //   println(s"Load successfully, leaf node count: ${tree.idCodeMap.size}, " +
 //     s"nonLeafOffset: ${tree.nonLeafOffset}")
  }

  private def loadData(path: String, tree: DistTree): Unit = {
    val kBatchSize = 500
    val kvData = mutable.HashMap.empty[String, Array[Byte]]
    val keys = new ArrayBuffer[String]()
    val values = new ArrayBuffer[Array[Byte]]()
    Using(new DataInputStream(new BufferedInputStream(new FileInputStream(path)))) { input =>
      while (input.available() > 0) {
        val num = input.readInt()
        val buf = new Array[Byte](num)
        // input.read(buf, 0, num)
        input.readFully(buf)
        val item = KVItem.parseFrom(buf)
        keys += item.key.toString(encoding.name())
        values += item.value.toByteArray
        if (keys.length >= kBatchSize) {
          multiPut(kvData, keys, values)
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
      multiPut(kvData, keys, values)
    }
    tree.kvData = kvData

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

  private def multiPut(
      map: mutable.HashMap[String, Array[Byte]],
      keys: ArrayBuffer[String],
      values: ArrayBuffer[Array[Byte]]): Unit = {

    keys.zip(values) foreach {
      case (k: String, v: Array[Byte]) => map(k) = v
    }
  }
}
