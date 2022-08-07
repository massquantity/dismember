package com.mass.retrieval.dr

import com.mass.dr.dataset.LocalDataSet
import com.mass.dr.model.{DeepRetrieval, LayerModel, MappingOp, RerankModel}
import com.mass.dr.optim.LocalOptimizer
import com.mass.scalann.utils.Property
import com.mass.scalann.utils.Property.getOrStop
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

object DRTrainDeepModel {

  case class Params(drConfFile: String = "fromResource")

  def main(args: Array[String]): Unit = {
    Logger.getLogger("com.mass").setLevel(Level.INFO)

    val defaultParams = Params()
    val parser = new OptionParser[Params]("TrainDeepModel") {
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
    val conf = Property.readConf(params.drConfFile, "model", "deep_retrieval", print = true)
    Property.configLocal(conf)

    val dataPath = getOrStop(conf, "data_path")
    val modelPath = getOrStop(conf, "model_path")
    val trainBatchSize = getOrStop(conf, "train_batch_size").toInt
    val evalBatchSize = getOrStop(conf, "eval_batch_size").toInt
    val numLayer = getOrStop(conf, "num_layer").toInt
    val numNode = getOrStop(conf, "num_node").toInt
    val numPathPerItem = getOrStop(conf, "num_path_per_item").toInt
    val embedSize = getOrStop(conf, "embed_size").toInt
    val learningRate = getOrStop(conf, "learning_rate").toDouble
    val numEpoch = getOrStop(conf, "epoch_num").toInt
    val numSampled = getOrStop(conf, "num_sampled").toInt
    val topk = getOrStop(conf, "topk_number").toInt
    val beamSize = getOrStop(conf, "beam_size").toInt
    val progressInterval = getOrStop(conf, "show_progress_interval").toInt
    val seqLen = getOrStop(conf, "seq_len").toInt
    val minSeqLen = getOrStop(conf, "min_seq_len").toInt
    val splitRatio = getOrStop(conf, "split_ratio").toDouble
    val initMapping = getOrStop(conf, "initialize_mapping").toBoolean
    val mappingPath = getOrStop(conf, "mapping_path")

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

    val layerModel = LayerModel(dataset.numItem, numNode, numLayer, seqLen, embedSize)
    val reRankModel = RerankModel(dataset.numItem, seqLen, embedSize)
    val optimizer = LocalOptimizer(
      dataset = dataset,
      layerModel = layerModel,
      reRankModel = reRankModel,
      numEpoch = numEpoch,
      numLayer = numLayer,
      learningRate = learningRate,
      numSampled = numSampled,
      embedSize = embedSize,
      topk = topk,
      beamSize = beamSize,
      progressInterval = progressInterval,
      reRankEpoch = None
    )
    optimizer.optimize()

    val mappingOp = MappingOp(dataset.itemIdMapping, dataset.itemPathMapping)
    val drModel = DeepRetrieval(
      layerModel = layerModel,
      reRankModel = reRankModel,
      numItem = dataset.numItem,
      numNode = numNode,
      numLayer = numLayer,
      seqLen = seqLen,
      embedSize = embedSize
    )
    DeepRetrieval.saveModel(drModel, modelPath)
    recommend(drModel, mappingOp)
  }

  def recommend(drModel: DeepRetrieval, mappingOp: MappingOp): Unit = {
    val sequence = Seq(0, 0, 2126, 204, 3257, 3439, 996, 1681, 3438, 1882)
    val rec = drModel.recommend(sequence, topk = 3, beamSize = 20, mappingOp)
    println(s"Recommendation result: $rec")

    (1 to 10).foreach(_ => drModel.recommend(sequence, topk = 10, beamSize = 20, mappingOp))
    val start = System.nanoTime()
    (1 to 100).foreach(_ => drModel.recommend(sequence, topk = 10, beamSize = 20, mappingOp))
    val end = System.nanoTime()
    println(f"Average recommend time: ${(end - start) / 100 / 1e6d}%.4fms")
  }
}
