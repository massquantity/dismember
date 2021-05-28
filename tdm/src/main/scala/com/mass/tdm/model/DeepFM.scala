package com.mass.tdm.model

import scala.reflect.ClassTag

import com.mass.sparkdl.Module
import com.mass.sparkdl.nn.{Add, FM, Graph, Input, Linear, Embedding, ReLU, Reshape}
import com.mass.sparkdl.tensor.TensorNumeric
import com.mass.tdm.operator.TDMOp

object DeepFM {

  def buildModel[@specialized(Float, Double) T: ClassTag](featSeqLen: Int,
      embedSize: Int, paddingIdx: Int)(implicit ev: TensorNumeric[T]): Module[T] = {

    val numIndex = (math.pow(2, TDMOp.tree.getMaxLevel) - 1).toInt
    val input = Input[T]()
    val embedding = Embedding[T](numIndex, embedSize, paddingIdx = paddingIdx)
      .inputs(input)
    // FM
    val fm = FM[T]()
      .inputs(embedding)
    // DNN
    val embeddingReshape = Reshape[T](Array(featSeqLen * embedSize))
      .inputs(embedding)
    val linear = Linear[T](featSeqLen * embedSize, featSeqLen)
      .inputs(embeddingReshape)
    val relu = ReLU[T]()
      .inputs(linear)
    val linear2 = Linear[T](featSeqLen, 1)
      .inputs(relu)
    //  val relu2 = ReLU[Float]().inputs(linear2)
    val add = Add[T]()
      .inputs(fm, linear2)

    Graph[T](input, add)
  }
}
