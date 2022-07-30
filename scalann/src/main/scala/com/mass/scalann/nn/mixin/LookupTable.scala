package com.mass.scalann.nn.mixin

import scala.collection.mutable.ListBuffer

import com.mass.scalann.tensor.{Tensor, TensorNumeric}

trait LookupTable[T] {

  protected val zeroArray: Array[T]

  protected def padWeight(weight: Tensor[T], paddingIdx: Int): Unit = {
    if (paddingIdx >= 0 && paddingIdx < weight.size(0)) {
      weight.select(0, paddingIdx).zero()
    }
  }

  protected def embeddingLookup(
      numElem: Int,
      indexData: Array[Int],
      indexOffset: Int,
      outputData: Array[T],
      outputOffset: Int,
      weightData: Array[T],
      weightOffset: Int,
      embedSize: Int,
      numIndex: Int,
      paddingIdx: Int): Unit = {

    val invalidIndices = ListBuffer.empty[String]
    var i = 0
    while (i < numElem) {
      val index = indexData(i + indexOffset)
      val _outputOffset = i * embedSize + outputOffset
      if (index == paddingIdx) {
        System.arraycopy(zeroArray, 0, outputData, _outputOffset, embedSize)
      } else if (index >= 0 && index < numIndex && index != paddingIdx) {
        val offset1 = index * embedSize + weightOffset
      //  println(weightData.length, offset1, outputData.length, _outputOffset, embedSize)
        System.arraycopy(weightData, offset1, outputData, _outputOffset, embedSize)
      } else {
        invalidIndices += s"\t ${i / embedSize}    ${i % embedSize}     $index"
      }
      i += 1
    }

    if (invalidIndices.nonEmpty) {
      throw new ArrayIndexOutOfBoundsException(
        s"\n${getClass.getSimpleName} --- embeddingLookup failed, " +
          s"valid index range is [0, $numIndex), but got: \n\trow column index\n" +
          s"${invalidIndices.mkString("\n")}"
      )
    }
  }

  protected def updateEmbeddings(
      numElem: Int,
      indexData: Array[Int],
      indexOffset: Int,
      gradOutputData: Array[T],
      gradOutputOffset: Int,
      gradWeightData: Array[T],
      gradWeightOffset: Int,
      embedSize: Int,
      paddingIdx: Int,
      scaleW: Double)(implicit ev: TensorNumeric[T]): Unit = {

    var i = 0
    while (i < numElem) {
      val index = indexData(i + indexOffset)
      if (index != paddingIdx) {
        val offset1 = i * embedSize + gradOutputOffset
        val offset2 = index * embedSize + gradWeightOffset
        ev.axpy(
          embedSize,
          ev.fromType(scaleW),
          gradOutputData,
          offset1,
          1,
          gradWeightData,
          offset2,
          1)
      }
      i += 1
    }
  }
}
