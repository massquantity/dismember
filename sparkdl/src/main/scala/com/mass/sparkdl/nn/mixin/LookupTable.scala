package com.mass.sparkdl.nn.mixin

import scala.collection.mutable.ListBuffer

import com.mass.sparkdl.tensor.{Tensor, TensorNumeric}

trait LookupTable[T] {

  protected def padWeight(weight: Tensor[T], paddingIdx: Int): Unit = {
    if (paddingIdx >= 0 && paddingIdx < weight.size(0)) {
      weight.select(0, paddingIdx).zero()
    }
  }

  protected def embeddingLookup(
      indexData: Array[Int],
      indexOffset: Int,
      outputData: Array[T],
      outputOffset: Int,
      weightData: Array[T],
      weightOffset: Int,
      embedSize: Int,
      numIndex: Int): Unit = {

    val invalidIndices = ListBuffer.empty[String]
    val numElem = indexData.length
    var i = 0
    while (i < numElem) {
      val index = indexData(i + indexOffset)
      if (index >= 0 && index < numIndex) {
        val offset1 = indexData(i + indexOffset) * embedSize + weightOffset
        val offset2 = i * embedSize + outputOffset
        System.arraycopy(weightData, offset1, outputData, offset2, embedSize)
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
      indexData: Array[Int],
      indexOffset: Int,
      gradOutputData: Array[T],
      gradOutputOffset: Int,
      gradWeightData: Array[T],
      gradWeightOffset: Int,
      embedSize: Int,
      paddingIdx: Int,
      scaleW: Double)(implicit ev: TensorNumeric[T]): Unit = {

    val numElem = indexData.length
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
