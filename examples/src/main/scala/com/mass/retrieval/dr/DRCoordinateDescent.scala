package com.mass.retrieval.dr

import com.mass.dr.dataset.LocalDataSet
import com.mass.dr.model.{DeepRetrieval, MappingOp}
import com.mass.dr.optim.CoordinateDescent
import com.mass.scalann.utils.Property
import com.mass.scalann.utils.Property.getOrStop
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

object DRCoordinateDescent {

  case class Params(drConfFile: String = "fromResource")

  def main(args: Array[String]): Unit = {
    Logger.getLogger("com.mass").setLevel(Level.INFO)

    val defaultParams = Params()
    val parser = new OptionParser[Params]("CoordinateDescent") {
      opt[String]("drConfFile")
        .text(s"Deep Retrieval config file path, default path is `deep_retrieval.conf` from resource folder")
        .action((x, c) => c.copy(drConfFile = x))
    }

    parser.parse(args, defaultParams) match {
      case Some(params) => run(params)
      case _ => sys.exit(1)
    }
  }

  def run(params: Params): Unit = {
    val conf = Property.readConf(params.drConfFile, "cd", "deep_retrieval", print = true)
    Property.configLocal(conf)

    val dataPath = getOrStop(conf, "data_path")
    val modelPath = getOrStop(conf, "model_path")
    val mappingPath = getOrStop(conf, "mapping_path")
    val trainBatchSize = getOrStop(conf, "train_batch_size").toInt
    val evalBatchSize = getOrStop(conf, "eval_batch_size").toInt
    val numLayer = getOrStop(conf, "num_layer").toInt
    val numNode = getOrStop(conf, "num_node").toInt
    val numPathPerItem = getOrStop(conf, "num_path_per_item").toInt
    val seqLen = getOrStop(conf, "seq_len").toInt
    val minSeqLen = getOrStop(conf, "min_seq_len").toInt
    val splitRatio = getOrStop(conf, "split_ratio").toDouble
    val initMapping = getOrStop(conf, "initialize_mapping").toBoolean
    val numCandidatePath = getOrStop(conf, "candidate_path_num").toInt
    val numIteration = getOrStop(conf, "iteration_num").toInt
    val decayFactor = getOrStop(conf, "decay_factor").toDouble
    val penaltyFactor = getOrStop(conf, "penalty_factor").toDouble
    val penaltyPolyOrder = getOrStop(conf, "penalty_poly_order").toInt
    val trainMode = getOrStop(conf, "train_mode")

    val dataset = LocalDataSet(
      numLayer = numLayer,
      numNode = numNode,
      numPathPerItem = numPathPerItem,
      trainBatchSize = trainBatchSize,
      evalBatchSize = evalBatchSize,
      seqLen = seqLen,
      minSeqLen = minSeqLen,
      dataPath = dataPath,
      mappingPath = mappingPath,
      initMapping = initMapping,
      splitRatio = splitRatio,
      delimiter = ","
    )

    val drModel = DeepRetrieval.loadModel(modelPath)
    val coo = CoordinateDescent(
      dataset = dataset,
      batchSize = trainBatchSize,
      numIteration = numIteration,
      numCandidatePath = numCandidatePath,
      numPathPerItem = numPathPerItem,
      numLayer = numLayer,
      numNode = numNode,
      decayFactor = decayFactor,
      penaltyFactor = penaltyFactor,
      penaltyPolyOrder = penaltyPolyOrder
    )
    val itemPathMapping = coo.optimize(drModel.layerModel, trainMode)
    MappingOp.writeMapping(mappingPath, dataset.itemIdMapping, itemPathMapping)
  }
}
