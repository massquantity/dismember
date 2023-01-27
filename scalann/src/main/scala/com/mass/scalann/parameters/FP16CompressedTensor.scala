package com.mass.scalann.parameters

import java.nio.ByteBuffer

import scala.reflect.{classTag, ClassTag}

import com.mass.scalann.tensor.Tensor
import com.mass.scalann.utils.Engine

private[scalann] class FP16CompressedTensor[T: ClassTag](
    buffer: Array[Byte],
    bufferOffset: Int,
    bufferLength: Int
) extends CompressedTensor[T] {

  def this(tensor: Tensor[T]) = {
    this(new Array[Byte](tensor.nElement() * 2), 0, tensor.nElement() * 2)
    compress(tensor)
  }

  def this(length: Int) = this(new Array[Byte](length * 2), 0, length * 2)

  def this(bytes: ByteBuffer) = this(bytes.array(), bytes.position(), bytes.remaining())

  require(bufferLength % 2 == 0 && bufferOffset + bufferLength <= buffer.length)

  override def bytes(): ByteBuffer = bytes(0, bufferLength / 2)

  override def bytes(offset: Int, length: Int): ByteBuffer = {
    require(offset >= 0 && length > 0 && (offset + length) * 2 <= bufferLength)
    if (classTag[T] == classTag[Double]) {
      ByteBuffer.wrap(buffer, offset * 2 + bufferOffset, length * 2)
    } else if (classTag[T] == classTag[Float]) {
      ByteBuffer.wrap(buffer, offset * 2 + bufferOffset, length * 2)
    } else {
      throw new IllegalArgumentException
    }
  }

  override def compress(src: Tensor[T]): this.type = compress(0, src, 0, src.nElement())

  override def compress(offset: Int, src: Tensor[T], srcOffset: Int, length: Int): this.type = {
    require(
      src.isContiguous && offset >= 0 && srcOffset >= 0 &&
        srcOffset + length <= src.nElement() && offset + length <= bufferLength / 2
    )
    val tOffset = src.storageOffset() + srcOffset
    if (classTag[T] == classTag[Double]) {
      FP16CompressedTensor.toFP16(
        src.storage().array().asInstanceOf[Array[Double]],
        tOffset,
        buffer,
        bufferOffset + offset,
        length
      )
    } else if (classTag[T] == classTag[Float]) {
      FP16CompressedTensor.toFP16(
        src.storage().array().asInstanceOf[Array[Float]],
        tOffset,
        buffer,
        bufferOffset + offset,
        length
      )
    } else {
      throw new IllegalArgumentException
    }
    this
  }

  override def deCompress(tensor: Tensor[T]): Unit = deCompress(0, tensor, 0, bufferLength / 2)

  override def deCompress(srcOffset: Int, tensor: Tensor[T], tgtOffset: Int, length: Int): Unit = {
    require(
      srcOffset >= 0 && length > 0 && (srcOffset + length) * 2 <= bufferLength &&
        tgtOffset >= 0 && tgtOffset + length <= tensor.nElement()
    )
    require(tensor.isContiguous)
    if (classTag[T] == classTag[Double]) {
      val tdata = tensor.storage().array().asInstanceOf[Array[Double]]
      val toffset = tensor.storageOffset() + tgtOffset
      FP16CompressedTensor.fromFP16(
        buffer,
        srcOffset * 2 + bufferOffset,
        length * 2,
        tdata,
        toffset
      )
    } else if (classTag[T] == classTag[Float]) {
      val tdata = tensor.storage().array().asInstanceOf[Array[Float]]
      val toffset = tensor.storageOffset() + tgtOffset
      FP16CompressedTensor.fromFP16(
        buffer,
        srcOffset * 2 + bufferOffset,
        length * 2,
        tdata,
        toffset
      )
    } else {
      throw new IllegalArgumentException
    }
  }

  override def add(data: ByteBuffer): this.type = add(data, 0, bufferLength / 2)

  override def add(data: ByteBuffer, offset: Int, length: Int): this.type = {
    require(offset >= 0 && length > 0 && (offset + length) * 2 <= bufferLength)
    require(length * 2 == data.remaining())
    FP16CompressedTensor.add(
      buffer,
      offset * 2 + bufferOffset,
      data.array(),
      data.position(),
      data.remaining()
    )
    this
  }

  override def parAdd(data: ByteBuffer): this.type = add(data, 0, bufferLength / 2)

  override def parAdd(data: ByteBuffer, offset: Int, length: Int): this.type = {
    require(offset >= 0 && length > 0 && (offset + length) * 2 <= bufferLength)
    require(length * 2 == data.remaining())
    FP16CompressedTensor.parAdd(
      buffer,
      offset * 2 + bufferOffset,
      data.array(),
      data.position(),
      data.remaining()
    )
    this
  }
}

object FP16CompressedTensor {

  private def parAdd(
      left: Array[Byte],
      lOffset: Int,
      right: Array[Byte],
      rOffset: Int,
      length: Int
  ): Array[Byte] = {
    require(length % 2 == 0)
    require(lOffset + length <= left.length)
    require(rOffset + length <= right.length)

    val elementSize = length / 2
    val taskSize = elementSize / Engine.coreNumber()
    val extraSize = elementSize % Engine.coreNumber()
    val availableTask = if (taskSize == 0) extraSize else Engine.coreNumber()
    Engine.default.invokeAndWait(
      (0 until availableTask).map(tid =>
        () => {
          val start = tid * taskSize * 2 + math.min(extraSize, tid) * 2
          val end = (tid + 1) * taskSize * 2 + math.min(extraSize, tid + 1) * 2
          var i = start
          while (i < end) {
            val sum = toFloat(left(i + lOffset), left(i + lOffset + 1)) +
              toFloat(right(i + rOffset), right(i + rOffset + 1))
            val bytes = truncate(sum)
            left(i + lOffset) = (bytes >>> 24 & 0xff).toByte
            left(i + lOffset + 1) = (bytes >>> 16 & 0xff).toByte
            i += 2
          }
        }
      )
    )
    left
  }

  private def add(
      left: Array[Byte],
      lOffset: Int,
      right: Array[Byte],
      rOffset: Int,
      length: Int
  ): Array[Byte] = {
    require(length % 2 == 0)
    require(lOffset + length <= left.length)
    require(rOffset + length <= right.length)

    var i = 0
    while (i < length) {
      val sum = toFloat(left(i + lOffset), left(i + lOffset + 1)) +
        toFloat(right(i + rOffset), right(i + rOffset + 1))
      val bytes = truncate(sum)
      left(i + lOffset) = (bytes >>> 24 & 0xff).toByte
      left(i + lOffset + 1) = (bytes >>> 16 & 0xff).toByte

      i += 2
    }
    left
  }

  private[parameters] def toFP16(
      src: Array[Float],
      srcOffset: Int,
      tgt: Array[Byte],
      tgtOffset: Int,
      length: Int
  ): Array[Byte] = {
    require(srcOffset + length <= src.length)
    require(tgtOffset + length * 2 <= tgt.length)

    val taskSize = length / Engine.coreNumber()
    val extraSize = length % Engine.coreNumber()
    val availableTask = if (taskSize == 0) extraSize else Engine.coreNumber()
    Engine.default.invokeAndWait(
      (0 until availableTask).map(tid =>
        () => {
          val start = tid * taskSize + math.min(extraSize, tid)
          val end = (tid + 1) * taskSize + math.min(extraSize, tid + 1)
          var i = start
          while (i < end) {
            val bytes = truncate(src(i + srcOffset))
            tgt(tgtOffset + i * 2) = (bytes >>> 24 & 0xff).toByte
            tgt(tgtOffset + i * 2 + 1) = (bytes >>> 16 & 0xff).toByte
            i += 1
          }
        }
      )
    )
    tgt
  }

  private[parameters] def toFP16(
      src: Array[Double],
      srcOffset: Int,
      tgt: Array[Byte],
      tgtOffset: Int,
      length: Int
  ): Array[Byte] = {
    require(srcOffset + length <= src.length)
    require(tgtOffset + length * 2 <= tgt.length)

    val taskSize = length / Engine.coreNumber()
    val extraSize = length % Engine.coreNumber()
    val availableTask = if (taskSize == 0) extraSize else Engine.coreNumber()
    Engine.default.invokeAndWait(
      (0 until availableTask).map(tid =>
        () => {
          val start = tid * taskSize + math.min(extraSize, tid)
          val end = (tid + 1) * taskSize + math.min(extraSize, tid + 1)
          var i = start
          while (i < end) {
            val bytes = truncate(src(i + srcOffset).toFloat)
            tgt(tgtOffset + i * 2) = (bytes >>> 24 & 0xff).toByte
            tgt(tgtOffset + i * 2 + 1) = (bytes >>> 16 & 0xff).toByte
            i += 1
          }
        }
      )
    )
    tgt
  }

  private[parameters] def fromFP16(
      fp16: Array[Byte],
      fp16Offset: Int,
      fp16Length: Int,
      target: Array[Float],
      targetOffset: Int
  ): Unit = {
    require(fp16Length % 2 == 0)
    require(fp16Length + fp16Offset <= fp16.length)
    require(fp16Length / 2 + targetOffset <= target.length)

    val targetLength = fp16Length / 2
    val taskSize = targetLength / Engine.coreNumber()
    val extraSize = targetLength % Engine.coreNumber()
    val availableTask = if (taskSize == 0) extraSize else Engine.coreNumber()
    Engine.default.invokeAndWait(
      (0 until availableTask).map(tid =>
        () => {
          val start = tid * taskSize * 2 + math.min(extraSize, tid) * 2
          val end = (tid + 1) * taskSize * 2 + math.min(extraSize, tid + 1) * 2
          var i = start
          while (i < end && i < fp16Length + fp16Offset) {
            target(i / 2 + targetOffset) = toFloat(fp16(i + fp16Offset), fp16(i + 1 + fp16Offset))
            i += 2
          }
        }
      )
    )
  }

  private[parameters] def fromFP16(
      fp16: Array[Byte],
      fp16Offset: Int,
      fp16Length: Int,
      target: Array[Double],
      targetOffset: Int
  ): Unit = {
    require(fp16Length % 2 == 0)
    require(fp16Length + fp16Offset <= fp16.length)
    require(fp16Length / 2 + targetOffset <= target.length)

    val targetLength = fp16Length / 2
    val taskSize = targetLength / Engine.coreNumber()
    val extraSize = targetLength % Engine.coreNumber()
    val availableTask = if (taskSize == 0) extraSize else Engine.coreNumber()
    Engine.default.invokeAndWait(
      (0 until availableTask).map(tid =>
        () => {
          val start = tid * taskSize * 2 + math.min(extraSize, tid) * 2
          val end = (tid + 1) * taskSize * 2 + math.min(extraSize, tid + 1) * 2
          var i = start
          while (i < end && i < fp16Length + fp16Offset) {
            target(i / 2 + targetOffset) = toFloat(fp16(i + fp16Offset), fp16(i + 1 + fp16Offset))
            i += 2
          }
        }
      )
    )
  }

  @inline
  private def truncate(value: Float): Int = {
    java.lang.Float.floatToRawIntBits(value) & 0xffff0000
  }

  @inline
  private def toFloat(byte1: Byte, byte2: Byte): Float = {
    java.lang.Float.intBitsToFloat(byte1 << 24 | byte2 << 16 & 0x00ff0000)
  }
}
