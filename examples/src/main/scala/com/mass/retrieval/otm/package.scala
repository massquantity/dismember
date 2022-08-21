package com.mass.retrieval

import com.mass.otm.model.OTM
import com.mass.scalann.utils.Engine
import com.mass.scalann.utils.Property.getOrStop

package object otm {

  sealed trait Params

  case class TrainModelParams(
    dataPath: String,
    modelPath: String,
    deepModelName: String,
    useMask: Boolean,
    numThreads: Int,
    trainBatchSize: Int,
    evalBatchSize: Int,
    embedSize: Int,
    learningRate: Double,
    numEpoch: Int,
    topk: Int,
    beamSize: Int,
    progressInterval: Int,
    seqLen: Int,
    minSeqLen: Int,
    splitRatio: Double,
    leafInitMode: String,
    initMapping: Boolean,
    mappingPath: String,
    labelNum: Int,
    targetMode: String,
    seed: Long
  ) extends Params

  case class TreeConstructionParams(
    dataPath: String,
    modelPath: String,
    mappingPath: String,
    deepModelName: String,
    useMask: Boolean,
    gap: Int,
    labelNum: Int,
    seqLen: Int,
    minSeqLen: Int,
    splitRatio: Double,
    numThreads: Int
  ) extends Params

  def getParameters(conf: Map[String, String], mode: String): Params = {
    val deepModelName = getOrStop(conf, "deep_model").toLowerCase
    val useMask = if (deepModelName == "din") true else false
    mode match {
      case "model" =>
        TrainModelParams(
          getOrStop(conf, "data_path"),
          getOrStop(conf, "model_path"),
          deepModelName,
          useMask,
          Engine.coreNumber(),
          getOrStop(conf, "train_batch_size").toInt,
          getOrStop(conf, "eval_batch_size").toInt,
          getOrStop(conf, "embed_size").toInt,
          getOrStop(conf, "learning_rate").toDouble,
          getOrStop(conf, "epoch_num").toInt,
          getOrStop(conf, "topk_number").toInt,
          getOrStop(conf, "beam_size").toInt,
          getOrStop(conf, "show_progress_interval").toInt,
          getOrStop(conf, "seq_len").toInt,
          getOrStop(conf, "min_seq_len").toInt,
          getOrStop(conf, "split_ratio").toDouble,
          getOrStop(conf, "leaf_init_mode"),
          getOrStop(conf, "initialize_mapping").toBoolean,
          getOrStop(conf, "mapping_path"),
          getOrStop(conf, "label_num").toInt,
          getOrStop(conf, "target_mode"),
          getOrStop(conf, "seed").toLong
        )
      case "tree" =>
        TreeConstructionParams(
          getOrStop(conf, "data_path"),
          getOrStop(conf, "model_path"),
          getOrStop(conf, "mapping_path"),
          deepModelName,
          useMask,
          getOrStop(conf, "gap").toInt,
          getOrStop(conf, "label_num").toInt,
          getOrStop(conf, "seq_len").toInt,
          getOrStop(conf, "min_seq_len").toInt,
          getOrStop(conf, "split_ratio").toDouble,
          Engine.coreNumber()
        )
    }
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
