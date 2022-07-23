package com.mass.dr.model

import com.mass.dr.{paddingIdx, LayerModule}
import com.mass.sparkdl.nn._
import com.mass.sparkdl.nn.graphnn.Graph
import com.mass.sparkdl.tensor.Tensor
import com.mass.sparkdl.tensor.TensorNumeric.NumericDouble
import com.mass.sparkdl.utils.{Table, Util}

class LayerModel(
    val numItem: Int,
    val numNode: Int,
    val numLayer: Int,
    seqLen: Int,
    embedSize: Int) extends Serializable {
  require(numLayer >= 2, "number of layers must be at least 2")

  private[dr] val model = buildModel()
  lazy val parameters = model.adjustParameters()
  lazy val embedParams: Tensor[Double] = model.fetchModuleParameters("embedding", "weight")
  lazy val linearParams: IndexedSeq[Seq[Tensor[Double]]] = (1 to numLayer).map { i =>
    model.fetchModuleParameters(s"linear_$i", Seq("weight", "bias"))
  }

  private def buildModel(): Graph[Double] = {
    val input = Array.fill(numLayer)(Input())
    val embedIndex = numItem + numNode * (numLayer - 1)
    val embedding = EmbeddingShare(embedIndex, embedSize, paddingIdx)
      .setName("embedding")
      .inputs(input: _*)
    val output = Array.range(0, numLayer).map { d =>
      val flattenSize = (seqLen + d) * embedSize
      val flatten = Reshape(Array(flattenSize))
        .setName(s"flatten_${d + 1}")
        .inputs(Seq((embedding, d)))
      val linear = Linear(flattenSize, numNode)
        .setName(s"linear_${d + 1}")
        .inputs(flatten)
      linear
    }
    Graph(input, output)
  }

  def forward(input: Seq[Tensor[Double]]): Table = {
    val _input = Table.seq(input)
    model.forward(_input).toTable
  }

  def backward(input: Seq[Tensor[Double]], gradOutput: Table): Table = {
    val _input = Table.seq(input)
    model.backward(_input, gradOutput).toTable
  }

  def duplicateModels(num: Int): (Seq[LayerModule[Double]], Seq[Tensor[Double]]) = {
    // make the parameters compact
    model.adjustParameters()
    val weights: Array[Tensor[Double]] = Util.getAndClearWeightBias(model.parameters())
    // all models share same weight
    val models = (1 to num).map { _ =>
      val m = model.cloneModule()
      Util.putWeightBias(weights, m)
      Util.initGradWeightBias(weights, m)
      m
    }
    Util.putWeightBias(weights, model)
    Util.initGradWeightBias(weights, model)
    val gradients = models.map(_.adjustParameters()._2)
    (models, gradients)
  }

  def inference(inputSeq: Seq[Int], rank: Int): Array[Double] = {
    val output = Tensor[Double](numNode)
    val linearWeight = linearParams(rank).head
    val linearBias = linearParams(rank).last
    val embedArray = embedParams.storage().array()
    val inputData = inputSeq.flatMap { i =>
      if (i == paddingIdx) {
        Seq.fill(embedSize)(0.0)
      } else {
        embedArray.slice(i * embedSize, i * embedSize + embedSize)
      }
    }
    val inputEmbed = Tensor(inputData.toArray, Array(inputSeq.length * embedSize))
    output.addmv(0.0, 1.0, linearWeight, inputEmbed)
    output.add(linearBias)
    output.storage().array()
  }

  def inferenceWithWeight(): Tensor[Double] = ???
}

object LayerModel {

  def apply(
      numItem: Int,
      numNode: Int,
      numLayer: Int,
      seqLen: Int,
      embedSize: Int): LayerModel = {
    new LayerModel(
      numItem,
      numNode,
      numLayer,
      seqLen,
      embedSize
    )
  }
}
