package com.mass.tdm.model

import scala.math.Ordering

import com.mass.sparkdl.Module
import com.mass.sparkdl.tensor.Tensor
import com.mass.tdm.operator.TDMOp
import com.mass.tdm.paddingIdx
import com.mass.tdm.utils.Serialization
import com.mass.tdm.utils.Serialization.{saveEmbeddings, saveModel => sersaveModel}

class TDM(
    featSeqLen: Int,
    val embedSize: Int,
    deepModel: String) extends Serializable with Recommender {

  private[this] var dlModel: Module[Float] = _
  private val dlModelName = deepModel.toLowerCase
  lazy val useMask: Boolean = if (dlModelName == "din") true else false

  def getModel: Module[Float] = {
    require(dlModel != null, "\nThe deep model hasn't been built, " +
      "please use TDM(...) instead of new TDM(...) to build model")
    dlModel
  }

  def setModel(model: Module[Float]): Unit = {
    dlModel = model
  }

  private def buildModel(): TDM.this.type = {
    if (dlModelName == "deepfm") {
      dlModel = DeepFM.buildModel(featSeqLen, embedSize, paddingIdx)
      dlModel.setName("DeepFM")
    } else if (dlModelName == "din") {
      dlModel = DIN.buildModel[Float](embedSize, paddingIdx)
      dlModel.setName("DIN")
    } else {
      throw new IllegalArgumentException("deepModel name should DeepFM or DIN")
    }
    this
  }

  def predict(sequence: Array[Int], target: Int): Double = {
    val (innerId, _) = TDMOp.tree.idToCode(sequence ++ Seq(target))
    val tensor = Tensor(innerId.toArray, Array(1, sequence.length + 1))
    val logit = dlModel.forward(tensor).toTensor[Float].value()
    TDM.sigmoid(logit)
  }

  def recommend(sequence: Array[Int], topk: Int, candidateNum: Int = 20): Array[(Int, Double)] = {
    val dummyConsumed = Set.empty[Int]
    val recs = _recommend(sequence, dlModel, TDMOp.tree, candidateNum, useMask, dummyConsumed)
    // recs.sorted(Ordering.by[TreeNodePred, Float](_.pred)(Ordering[Float].reverse))
    recs.sortBy(_._2)(Ordering[Float].reverse).take(topk).map(i => (i._1, TDM.sigmoid(i._2)))
  }

  private def clearState(): Unit = {
    dlModel.clearState()
  }
}

object TDM {

  def apply(featSeqLen: Int, embedSize: Int, deepModel: String): TDM = {
    val tdm = new TDM(featSeqLen, embedSize, deepModel)
    tdm.buildModel()
  }

  def saveModel(modelPath: String, embedPath: String, model: TDM): Unit = {
    model.clearState()
    sersaveModel(modelPath, model.getModel)
    saveEmbeddings(embedPath, model.getModel, model.embedSize)
  }

  def loadModel(path: String): TDM = {
    val tdm = new TDM(0, 0, "DIN")
    tdm.setModel(Serialization.loadModel(path))
    tdm
  }

  def loadTree(treePbPath: String): Unit = {
    TDMOp.initTree(treePbPath)
  }

  @inline
  def sigmoid(logit: Float): Double = {
    1.0 / (1 + java.lang.Math.exp(-logit))
  }
}
