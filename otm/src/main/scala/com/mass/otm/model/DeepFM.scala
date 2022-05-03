package com.mass.otm.model

import scala.reflect.ClassTag

import com.mass.otm.{paddingIdx, DeepModel}
import com.mass.sparkdl.nn._
import com.mass.sparkdl.nn.graphnn.Graph
import com.mass.sparkdl.tensor.TensorNumeric

object DeepFM {

  def buildModel[@specialized(Float, Double) T: ClassTag](featSeqLen: Int, embedSize: Int, numIndex: Int)(
    implicit ev: TensorNumeric[T]
  ): DeepModel[T] = {
    val totalLen = featSeqLen + 1
    val inputItem = Input[T]()
    val inputSeq = Input[T]()

    val embedding = EmbeddingShare[T](numIndex, embedSize, paddingIdx = paddingIdx)
      .inputs(inputItem, inputSeq)
    val itemFlatten = Reshape[T](Array(embedSize))
      .inputs(Seq((embedding, 0)))
    val seqFlatten = Reshape[T](Array(featSeqLen * embedSize))
      .inputs(Seq((embedding, 1)))

    // FM
    val fmConcat = Concat[T]()
      .inputs(itemFlatten, seqFlatten)
    val fmFeature = Reshape[T](Array(totalLen, embedSize))
      .inputs(fmConcat)
    val fm = FM[T]()
      .inputs(fmFeature)
    // DNN
    val dnnFeature = Concat[T]()
      .inputs(itemFlatten, seqFlatten)
    val linear = Linear[T](totalLen * embedSize, totalLen)
      .inputs(dnnFeature)
    val relu = ReLU[T]()
      .inputs(linear)
    val linear2 = Linear[T](totalLen, 1)
      .inputs(relu)
    val add = Add[T]()
      .inputs(fm, linear2)

    Graph[T](Array(inputItem, inputSeq), Array(add))
  }
}
