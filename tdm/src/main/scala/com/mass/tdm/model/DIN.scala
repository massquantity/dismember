package com.mass.tdm.model

import scala.reflect.ClassTag

import com.mass.sparkdl.tensor.TensorNumeric
import com.mass.sparkdl.Module
import com.mass.sparkdl.nn._
import com.mass.tdm.operator.TDMOp

object DIN {

  def buildModel[@specialized(Float, Double) T: ClassTag](embedSize: Int, paddingIdx: Int)(
      implicit ev: TensorNumeric[T]): Module[T] = {

    val numIndex = (math.pow(2, TDMOp.tree.getMaxLevel) - 1).toInt
    val inputItem = Input[T]()
    val inputSeq = Input[T]()
    val inputMask = Input[T]()

    val embedding = EmbeddingShare[T](numIndex, embedSize, paddingIdx = paddingIdx)
      .inputs(inputItem, inputSeq)
    val item = Identity[T]()
      .inputs(Seq((embedding, 0)))
    val sequence = Identity[T]()
      .inputs(Seq((embedding, 1)))

    val attention = Attention[T](embedSize, useScale = true, linearTransform = false)
      .inputs(item, sequence, inputMask)
    val concat = Concat[T](flatten = true)
      .inputs(item, attention)

    val linear1 = Linear[T](2 * embedSize, embedSize)
      .inputs(concat)
    val relu = ReLU[T]()
      .inputs(linear1)
    val linear2 = Linear[T](embedSize, 1)
      .inputs(relu)

    Graph[T](Array(inputItem, inputSeq, inputMask), Array(linear2))
  }
}
