package com.mass.tdm.model

import scala.math.Ordering

import com.mass.sparkdl.Module
import com.mass.sparkdl.nn.{Add, BCECriterionWithLogits, FM, Graph, Linear, LookupTable, Reshape}
import com.mass.sparkdl.optim.Adam
import com.mass.sparkdl.tensor.Tensor
import com.mass.tdm.dataset.LocalDataSet
import com.mass.tdm.optim.LocalOptimizer
import com.mass.tdm.operator.TDMOp
import com.mass.tdm.tree.DistTree
import com.mass.tdm.utils.Serialization
import com.mass.tdm.utils.Serialization.{saveEmbeddings, saveModel => sersaveModel}

class TDM(
  //  numIndex: Int,
    featSeqLen: Int,
    val embedSize: Int
  //  lr: Double,
  //  numIteration: Int,
  //  progressInterval: Int = 1
    ) extends Serializable with Recommender {

  private[this] var dlModel: Module[Float] = _
  // @transient private[this] var tdmTree: DistTree = TDMOp.tree

  // def loadTree(pbFilePath: String): Unit = {
  //  tdmTree = DistTree(pbFilePath)
  // }

  def getModel: Module[Float] = {
    require(dlModel != null, "\nThe deep model hasn't been built, " +
      "please use TDM(...) instead of new TDM(...) to build model")
    dlModel
  }

  def setModel(model: Module[Float]): Unit = {
    dlModel = model
  }

  private def buildModel(): TDM.this.type = {
    val numIndex = (math.pow(2, TDMOp.tree.getMaxLevel) - 1).toInt
    val embedding = LookupTable[Float](numIndex, embedSize).inputs()
    val fm = FM[Float]().inputs(embedding)  // FM
    val embeddingReshape = Reshape[Float](Array(featSeqLen * embedSize)).inputs(embedding)
    val linear = Linear[Float](featSeqLen * embedSize, featSeqLen).inputs(embeddingReshape)
    val linear2 = Linear[Float](featSeqLen, 1).inputs(linear)  // DNN
    val add = Add[Float]().inputs(fm, linear2)
    dlModel = Graph[Float](embedding, add)
    this
  }

  /*
  def fit(dataset: LocalDataSet): this.type = {
    val optimMethod = new LocalOptimizer(
      model = dlModel,
      dataset = dataset,
      criterion = BCECriterionWithLogits(),
      optimMethod = Adam(learningRate = lr),
      numIteration = numIteration,
      progressInterval = progressInterval,
      topk = topk,
      candidateNum = candidateNum
    )
    optimMethod.optimize()
    this
  }
  */

  def predict(sequence: Array[Int], target: Int): Double = {
    val tensor = Tensor(TDMOp.tree.idToCode(sequence ++ Seq(target)),
      Array(1, sequence.length + 1))
    val logit = dlModel.forward(tensor).toTensor[Float].value()
    sigmoid(logit)
  }

  def recommend(sequence: Array[Int], topk: Int, candidateNum: Int = 20): Array[(Int, Double)] = {
    val recs = _recommend(sequence, dlModel, TDMOp.tree, candidateNum)
    // recs.sorted(Ordering.by[TreeNodePred, Float](_.pred)(Ordering[Float].reverse))
    recs.sortBy(_._2)(Ordering[Float].reverse).take(topk).map(i => (i._1, sigmoid(i._2)))
  }
}

object TDM {

  def apply(featSeqLen: Int, embedSize: Int): TDM = {
    val tdm = new TDM(featSeqLen, embedSize)
    tdm.buildModel()
  }

  def saveModel(modelPath: String, embedPath: String, model: TDM): Unit = {
    sersaveModel(modelPath, model.getModel)
    saveEmbeddings(embedPath, model.getModel, model.embedSize)
  }

  def loadModel(path: String): TDM = {
    val tdm = new TDM(0, 0)
    tdm.setModel(Serialization.loadModel(path))
    tdm
  }

  def loadTree(treePbPath: String): Unit = {
    TDMOp.initTree(treePbPath)
  }
}
