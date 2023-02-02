package com.mass.otm.model

import com.mass.otm.{paddingIdx, upperLog2, DeepModel}
import com.mass.tdm.utils.Serialization

class OTM(deepModel: DeepModel[Double], itemIdMapping: Map[Int, Int], useMask: Boolean)
    extends Serializable
    with CandidateSearcher {
  import OTM._

  val idItemMapping: Map[Int, Int] = itemIdMapping.map(_.swap)
  val leafLevel = upperLog2(itemIdMapping.size)

  def recommend(sequence: Seq[Int], topk: Int, beamSize: Int): Seq[(Int, Double)] = {
    val sequenceIds = sequence.map(itemIdMapping.getOrElse(_, paddingIdx))
    val candidates = beamSearch(sequenceIds, deepModel, leafLevel, beamSize, useMask)
    candidates
      .filter(i => idItemMapping.contains(i.id))
      .sortBy(_.score)(Ordering[Double].reverse)
      .take(topk)
      .map(i => (idItemMapping(i.id), sigmoid(i.score)))
  }
}

object OTM {

  def apply(
      deepModel: DeepModel[Double],
      itemIdMapping: Map[Int, Int],
      modelName: String
  ): OTM = {
    val useMask = if (modelName.toLowerCase == "din") true else false
    new OTM(deepModel, itemIdMapping, useMask)
  }

  val sigmoid = (logit: Double) => 1.0 / (1 + math.exp(-logit))

  def saveModel(
      modelPath: String,
      mappingPath: String,
      model: DeepModel[Double],
      itemIdMapping: Map[Int, Int]
  ): Unit = {
    model.clearState()
    Serialization.saveModel[Double](modelPath, model)
    Serialization.saveMapping(mappingPath, itemIdMapping)
  }

  def loadModel(
      modelPath: String,
      mappingPath: String,
      modelName: String
  ): OTM = {
    val name = modelName.toLowerCase
    require(name == "din" || name == "deepfm", "DeepModel should either be `DeepFM` or `DIN`")
    val deepModel = Serialization.loadModel[Double](modelPath)
    val itemIdMapping = Serialization.loadMapping(mappingPath)
    val useMask = if (name == "din") true else false
    new OTM(deepModel, itemIdMapping, useMask)
  }
}
