package com.mass.retrieval

import com.mass.dr.model.{DeepRetrieval, MappingOp}
import com.mass.scalann.utils.Property.getOrStop

package object dr {

  sealed trait Params

  case class TrainModelParams(
    dataPath: String,
    modelPath: String,
    mappingPath: String,
    trainBatchSize: Int,
    evalBatchSize: Int,
    numLayer: Int,
    numNode: Int,
    numPathPerItem: Int,
    embedSize: Int,
    learningRate: Double,
    numEpoch: Int,
    numSampled: Int,
    topk: Int,
    beamSize: Int,
    progressInterval: Int,
    seqLen: Int,
    minSeqLen: Int,
    splitRatio: Double,
    initMapping: Boolean
  ) extends Params

  case class CoordinateDescentParams(
    dataPath: String,
    modelPath: String,
    mappingPath: String,
    trainBatchSize: Int,
    evalBatchSize: Int,
    numLayer: Int,
    numNode: Int,
    numPathPerItem: Int,
    seqLen: Int,
    minSeqLen: Int,
    splitRatio: Double,
    initMapping: Boolean,
    numCandidatePath: Int,
    numIteration: Int,
    decayFactor: Double,
    penaltyFactor: Double,
    penaltyPolyOrder: Int,
    trainMode: String
  ) extends Params

  def getParameters(conf: Map[String, String], mode: String): Params = {
    mode match {
      case "model" =>
        TrainModelParams(
          getOrStop(conf, "data_path"),
          getOrStop(conf, "model_path"),
          getOrStop(conf, "mapping_path"),
          getOrStop(conf, "train_batch_size").toInt,
          getOrStop(conf, "eval_batch_size").toInt,
          getOrStop(conf, "num_layer").toInt,
          getOrStop(conf, "num_node").toInt,
          getOrStop(conf, "num_path_per_item").toInt,
          getOrStop(conf, "embed_size").toInt,
          getOrStop(conf, "learning_rate").toDouble,
          getOrStop(conf, "epoch_num").toInt,
          getOrStop(conf, "num_sampled").toInt,
          getOrStop(conf, "topk_number").toInt,
          getOrStop(conf, "beam_size").toInt,
          getOrStop(conf, "show_progress_interval").toInt,
          getOrStop(conf, "seq_len").toInt,
          getOrStop(conf, "min_seq_len").toInt,
          getOrStop(conf, "split_ratio").toDouble,
          getOrStop(conf, "initialize_mapping").toBoolean
        )

      case "cd" =>
        CoordinateDescentParams(
          getOrStop(conf, "data_path"),
          getOrStop(conf, "model_path"),
          getOrStop(conf, "mapping_path"),
          getOrStop(conf, "train_batch_size").toInt,
          getOrStop(conf, "eval_batch_size").toInt,
          getOrStop(conf, "num_layer").toInt,
          getOrStop(conf, "num_node").toInt,
          getOrStop(conf, "num_path_per_item").toInt,
          getOrStop(conf, "seq_len").toInt,
          getOrStop(conf, "min_seq_len").toInt,
          getOrStop(conf, "split_ratio").toDouble,
          getOrStop(conf, "initialize_mapping").toBoolean,
          getOrStop(conf, "candidate_path_num").toInt,
          getOrStop(conf, "iteration_num").toInt,
          getOrStop(conf, "decay_factor").toDouble,
          getOrStop(conf, "penalty_factor").toDouble,
          getOrStop(conf, "penalty_poly_order").toInt,
          getOrStop(conf, "train_mode")
      )
    }
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

  def showTime[T](block: => T, name: String): T = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    println(f"$name time: ${(t1 - t0) / 1e9d}%.4fs")
    result
  }
}
