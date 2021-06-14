package com.mass.tdm.model

import com.mass.sparkdl.Module
import com.mass.sparkdl.nn.{Add, Concat, EmbeddingShare, FM, Graph, Input, Linear, ReLU, Reshape}
import com.mass.tdm.operator.TDMOp

object DeepFM {

  def buildModel(featSeqLen: Int, embedSize: Int, paddingIdx: Int): Module[Float] = {

    val totalLen = featSeqLen + 1
    val numIndex = (math.pow(2, TDMOp.tree.getMaxLevel) - 1).toInt
    val inputItem = Input[Float]()
    val inputSeq = Input[Float]()

    val embedding = EmbeddingShare[Float](numIndex, embedSize, paddingIdx = paddingIdx)
      .inputs(inputItem, inputSeq)
    val itemFlatten = Reshape[Float](Array(embedSize))
      .inputs(Seq((embedding, 0)))
    val seqFlatten = Reshape[Float](Array(featSeqLen * embedSize))
      .inputs(Seq((embedding, 1)))

    // FM
    val fmConcat = Concat[Float]()
      .inputs(itemFlatten, seqFlatten)
    val fmFeature = Reshape[Float](Array(totalLen, embedSize))
      .inputs(fmConcat)
    val fm = FM[Float]()
      .inputs(fmFeature)
    // DNN
    val dnnFeature = Concat[Float]()
      .inputs(itemFlatten, seqFlatten)
    val linear = Linear[Float](totalLen * embedSize, totalLen)
      .inputs(dnnFeature)
    val relu = ReLU[Float]()
      .inputs(linear)
    val linear2 = Linear[Float](totalLen, 1)
      .inputs(relu)
    val add = Add[Float]()
      .inputs(fm, linear2)

    Graph[Float](Array(inputItem, inputSeq), Array(add))
  }
}
