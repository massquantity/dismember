package com.mass.tdm.model

import com.mass.scalann.Module
import com.mass.scalann.tensor.Tensor
import com.mass.tdm.operator.TDMOp
import com.mass.tdm.utils.Serialization

class TDM(dlModel: Module[Float], useMask: Boolean) extends Serializable with Recommender {

  def predict(sequence: Array[Int], target: Int): Double = {
    val (innerId, _) = TDMOp.tree.idToCode(sequence ++ Seq(target))
    val tensor = Tensor(innerId.toArray, Array(1, sequence.length + 1))
    val logit = dlModel.forward(tensor).toTensor[Float].value()
    TDM.sigmoid(logit)
  }

  def recommend(sequence: Array[Int], topk: Int, candidateNum: Int): Array[(Int, Double)] = {
    val dummyConsumed = Set.empty[Int]
    val recs = _recommend(sequence, dlModel, TDMOp.tree, candidateNum, useMask, dummyConsumed)
    // recs.sorted(Ordering.by[TreeNodePred, Float](_.pred)(Ordering[Float].reverse))
    recs.sortBy(_._2)(Ordering[Float].reverse).take(topk).map(i => (i._1, TDM.sigmoid(i._2)))
  }
}

object TDM {

  def apply(dlModel: Module[Float], modelName: String): TDM = {
    val useMask = if (modelName.toLowerCase == "din") true else false
    new TDM(dlModel, useMask)
  }

  def saveModel(
      modelPath: String,
      embedPath: String,
      model: Module[Float],
      embedSize: Int
  ): Unit = {
    model.clearState()
    Serialization.saveModel[Float](modelPath, model)
    Serialization.saveEmbeddings[Float](embedPath, model, embedSize)
  }

  def loadModel(modelPath: String, modelName: String): TDM = {
    val name = modelName.toLowerCase
    require(name == "din" || name == "deepfm", "DeepModel name should either be DeepFM or DIN")
    val dlModel = Serialization.loadModel[Float](modelPath)
    val useMask = if (name == "din") true else false
    new TDM(dlModel, useMask)
  }

  def loadTree(treePbPath: String): Unit = {
    TDMOp.initTree(treePbPath)
  }

  @inline
  def sigmoid(logit: Float): Double = {
    1.0 / (1 + java.lang.Math.exp(-logit))
  }
}
