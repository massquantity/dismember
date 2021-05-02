package org.apache.spark.storage

import java.lang.{Boolean => JBoolean}
import java.nio.ByteBuffer

import scala.reflect.ClassTag

import org.apache.spark.SparkEnv
import org.apache.spark.util.io.ChunkedByteBuffer

object BlockManagerWrapper {

  def putBytes(blockId: BlockId, bytes: ByteBuffer, level: StorageLevel): Unit = {
    require(bytes != null)
    putBytesFn(blockId, new ChunkedByteBuffer(bytes), level)
  }

  def getLocal(blockId: BlockId): Option[BlockResult] = {
    SparkEnv.get.blockManager.getLocalValues(blockId)
  }

  def putSingle(blockId: BlockId, value: Any, level: StorageLevel,
      tellMaster: Boolean = true): Unit = {
    SparkEnv.get.blockManager.putSingle(blockId, value, level, tellMaster)
  }

  def removeBlock(blockId: BlockId): Unit = {
    SparkEnv.get.blockManager.removeBlock(blockId)
  }

  def getLocalBytes(blockId: BlockId): Option[ByteBuffer] = {
    getLocalBytesFn(blockId)
  }

  def getLocalOrRemoteBytes(blockId: BlockId): Option[ByteBuffer] = {
    val maybeLocalBytes = getLocalBytesFn(blockId)
    if (maybeLocalBytes.isDefined) {
      maybeLocalBytes
    } else {
      SparkEnv.get.blockManager.getRemoteBytes(blockId).map(_.toByteBuffer)
    }
  }

  def unlock(blockId: BlockId): Unit = {
    val blockInfoManager = SparkEnv.get.blockManager.blockInfoManager
    if (blockInfoManager.get(blockId).isDefined) {
      unlockFn(blockId)
    }
  }

  private val getLocalBytesFn: BlockId => Option[ByteBuffer] = {
    val bmClass = classOf[BlockManager]
    val getLocalBytesMethod = bmClass.getMethod("getLocalBytes", classOf[BlockId])
    try {
      val blockDataClass = Class.forName("org.apache.spark.storage.BlockData")
      val toByteBufferMethod = blockDataClass.getMethod("toByteBuffer")
      blockId: BlockId =>
        getLocalBytesMethod.invoke(SparkEnv.get.blockManager, blockId)
          .asInstanceOf[Option[_]]
          .map(blockData => toByteBufferMethod.invoke(blockData).asInstanceOf[ByteBuffer])
    } catch {
      case _: ClassNotFoundException =>
        blockId: BlockId =>
          getLocalBytesMethod.invoke(SparkEnv.get.blockManager, blockId)
            .asInstanceOf[Option[ChunkedByteBuffer]]
            .map(_.toByteBuffer)
    }
  }

  private val putBytesFn: (BlockId, ChunkedByteBuffer, StorageLevel) => Unit = {
    val bmClass = classOf[BlockManager]
    val putBytesMethod =
      try {
        bmClass.getMethod("putBytes",
          classOf[BlockId], classOf[ChunkedByteBuffer], classOf[StorageLevel],
          classOf[Boolean], classOf[ClassTag[_]])
      } catch {
        case _: NoSuchMethodException =>
          bmClass.getMethod("putBytes",
            classOf[BlockId], classOf[ChunkedByteBuffer], classOf[StorageLevel],
            classOf[Boolean], classOf[Boolean], classOf[ClassTag[_]])
      }
    putBytesMethod.getParameterTypes.length match {
      case 5 =>
        (blockId: BlockId, bytes: ChunkedByteBuffer, level: StorageLevel) =>
          putBytesMethod.invoke(SparkEnv.get.blockManager,
            blockId, bytes, level, JBoolean.TRUE, null)
      case 6 =>
        (blockId: BlockId, bytes: ChunkedByteBuffer, level: StorageLevel) =>
          putBytesMethod.invoke(SparkEnv.get.blockManager,
            blockId, bytes, level, JBoolean.TRUE, JBoolean.FALSE, null)
    }
  }

  private val unlockFn: BlockId => Unit = {
    val bimClass = classOf[BlockInfoManager]
    val unlockMethod =
    try {
      bimClass.getMethod("unlock", classOf[BlockId])
    } catch {
      case _: NoSuchMethodException =>
        bimClass.getMethod("unlock", classOf[BlockId], classOf[Option[_]])
    }
    unlockMethod.getParameterTypes.length match {
      case 1 =>
        (blockId: BlockId) =>
          unlockMethod.invoke(SparkEnv.get.blockManager.blockInfoManager, blockId)
      case 2 =>
        (blockId: BlockId) =>
          unlockMethod.invoke(SparkEnv.get.blockManager.blockInfoManager, blockId, None)
    }
  }
}
