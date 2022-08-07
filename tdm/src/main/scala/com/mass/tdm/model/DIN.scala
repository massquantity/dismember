package com.mass.tdm.model

import scala.reflect.ClassTag

import com.mass.scalann.tensor.TensorNumeric
import com.mass.scalann.Module
import com.mass.scalann.nn._
import com.mass.scalann.nn.graphnn.Graph
import com.mass.tdm.operator.TDMOp
import com.mass.tdm.paddingIdx

object DIN {

  def buildModel[@specialized(Float, Double) T: ClassTag](embedSize: Int)(
      implicit ev: TensorNumeric[T]): Module[T] = {

    val numIndex = (math.pow(2, TDMOp.tree.getMaxLevel + 1) - 1).toInt
    val inputItem = Input[T]()
    val inputSeq = Input[T]()
    val inputMask = Input[T]()

    val embedding = EmbeddingShare[T](numIndex, embedSize, paddingIdx)
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

    Graph[T](Seq(inputItem, inputSeq, inputMask), Seq(linear2))
  }
}
