package com.mass.retrieval.otm

import com.mass.otm.dataset.LocalDataSet
import com.mass.otm.model.{DeepFM, DIN, OTM}
import com.mass.otm.optim.LocalOptimizer
import com.mass.scalann.utils.{Engine, Property}
import com.mass.scalann.utils.Property.getOrStop
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

object OTMTrainDeepModel {

  case class Params(otmConfFile: String = "fromResource")

  def main(args: Array[String]): Unit = {
    Logger.getLogger("com.mass").setLevel(Level.INFO)

    val defaultParams = Params()
    val parser = new OptionParser[Params]("TrainDeepModel") {
      opt[String]("otmConfFile")
        .text(s"OTM config file path, default path is `otm.conf` from resource folder")
        .action((x, c) => c.copy(otmConfFile = x))
    }

    parser.parse(args, defaultParams) match {
      case Some(params) => run(params)
      case _ => sys.exit(1)
    }
  }

  def run(params: Params): Unit = {
    val conf = Property.readConf(params.otmConfFile, "model", "otm", print = true)
    Property.configLocal(conf)

    val dataPath = getOrStop(conf, "data_path")
    val modelPath = getOrStop(conf, "model_path")
    val deepModelName = getOrStop(conf, "deep_model").toLowerCase
    val useMask = if (deepModelName == "din") true else false
    val numThreads = Engine.coreNumber()
    val trainBatchSize = getOrStop(conf, "train_batch_size").toInt
    val evalBatchSize = getOrStop(conf, "eval_batch_size").toInt
    val embedSize = getOrStop(conf, "embed_size").toInt
    val learningRate = getOrStop(conf, "learning_rate").toDouble
    val numEpoch = getOrStop(conf, "epoch_num").toInt
    val topk = getOrStop(conf, "topk_number").toInt
    val beamSize = getOrStop(conf, "beam_size").toInt
    val progressInterval = getOrStop(conf, "show_progress_interval").toInt
    val seqLen = getOrStop(conf, "seq_len").toInt
    val minSeqLen = getOrStop(conf, "min_seq_len").toInt
    val splitRatio = getOrStop(conf, "split_ratio").toDouble
    val leafInitMode = getOrStop(conf, "leaf_init_mode")
    val initMapping = getOrStop(conf, "initialize_mapping").toBoolean
    val mappingPath = getOrStop(conf, "mapping_path")
    val labelNum = getOrStop(conf, "label_num").toInt
    val targetMode = getOrStop(conf, "target_mode")
    val seed = getOrStop(conf, "seed").toLong
    require(trainBatchSize >= numThreads)

    val dataset = LocalDataSet(
      dataPath = dataPath,
      seqLen = seqLen,
      minSeqLen = minSeqLen,
      splitRatio = splitRatio,
      leafInitMode = leafInitMode,
      initMapping = initMapping,
      mappingPath = mappingPath,
      labelNum = labelNum,
      seed = seed,
    )
    val deepModel =
      if (deepModelName == "din") {
        DIN.buildModel[Double](embedSize, dataset.numTreeNode)
      } else if (deepModelName == "deepfm") {
        DeepFM.buildModel[Double](seqLen, embedSize, dataset.numTreeNode)
      } else {
        throw new IllegalArgumentException("DeepModel name should either be DeepFM or DIN")
      }

    val optimizer = LocalOptimizer(
      deepModel = deepModel,
      dataset = dataset,
      targetMode = targetMode,
      numEpoch = numEpoch,
      totalTrainBatchSize = trainBatchSize,
      totalEvalBatchSize = evalBatchSize,
      learningRate = learningRate,
      beamSize = beamSize,
      topk = topk,
      seqLen = seqLen,
      useMask = useMask,
      progressInterval = progressInterval
    )
    optimizer.optimize()

    OTM.saveModel(modelPath, mappingPath, deepModel, dataset.itemIdMapping)
    val otm = OTM(deepModel, dataset.itemIdMapping, deepModelName)
    recommend(otm)
  }

  def recommend(otmModel: OTM): Unit = {
    val sequence = Seq(0, 0, 2126, 204, 3257, 3439, 996, 1681, 3438, 1882)
    val rec = otmModel.recommend(sequence, topk = 3, beamSize = 20)
    println(s"Recommendation result: $rec")

    (1 to 10).foreach(_ => otmModel.recommend(sequence, topk = 10, beamSize = 20))
    val start = System.nanoTime()
    (1 to 100).foreach(_ => otmModel.recommend(sequence, topk = 10, beamSize = 20))
    val end = System.nanoTime()
    println(f"Average recommend time: ${(end - start) / 100 / 1e6d}%.4fms")
  }
}
