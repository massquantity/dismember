package com.mass.retrieval

import com.mass.scalann.utils.Engine
import com.mass.scalann.utils.Property.getOrStop
import com.mass.tdm.model.TDM

package object tdm {

  sealed trait Params

  case class InitTreeParams(
    seqLen: Int,
    minSeqLen: Int,
    splitForEval: Boolean,
    splitRatio: Double,
    dataPath: String,
    trainPath: String,
    evalPath: Option[String],
    statPath: String,
    leafIdPath: String,
    treePbPath: String,
    userConsumedPath: Option[String]
  ) extends Params

  case class TrainModelParams(
    deepModelName: String,
    seqLen: Int,
    useMask: Boolean,
    totalBatchSize: Int,
    totalEvalBatchSize: Int,
    layerNegCounts: String,
    withProb: Boolean,
    startSampleLevel: Int,
    tolerance: Int,
    numThreads: Int,
    trainPath: String,
    evalPath: String,
    pbFilePath: String,
    userConsumedPath: String,
    embedSize: Int,
    learningRate: Double,
    numIteration: Int,
    progressInterval: Int,
    topk: Int,
    candidateNum: Int,
    modelPath: String,
    embedPath: String
  ) extends Params

  case class ClusterTreeParams(
    embedPath: String,
    parallel: Boolean,
    numThreads: Int,
    clusterIterNum: Int,
    outputTreePath: String,
    clusterType: String
  ) extends Params

  def getParameters(conf: Map[String, String], mode: String): Params = {
    mode match {
      case "init" =>
        InitTreeParams(
          getOrStop(conf, "seq_len").toInt,
          getOrStop(conf, "min_seq_len").toInt,
          conf.getOrElse("split_for_eval", "true").toBoolean,
          conf.getOrElse("split_ratio", "0.8").toDouble,
          getOrStop(conf, "data_path"),
          getOrStop(conf, "train_path"),
          conf.get("eval_path"),
          getOrStop(conf, "stat_path"),
          getOrStop(conf, "leaf_id_path"),
          getOrStop(conf, "tree_protobuf_path"),
          conf.get("user_consumed_path")
        )
      case "model" =>
        val deepModelName = getOrStop(conf, "deep_model").toLowerCase
        val useMask = if (deepModelName == "din") true else false
        TrainModelParams(
          deepModelName,
          getOrStop(conf, "seq_len").toInt,
          useMask,
          getOrStop(conf, "total_batch_size").toInt,
          getOrStop(conf, "total_eval_batch_size").toInt,
          getOrStop(conf, "layer_negative_counts"),
          conf.getOrElse("sample_with_probability", "true").toBoolean,
          conf.getOrElse("start_sample_level", "1").toInt,
          conf.getOrElse("sample_tolerance", "20").toInt,
          Engine.coreNumber(),
          getOrStop(conf, "train_path"),
          getOrStop(conf, "eval_path"),
          getOrStop(conf, "tree_protobuf_path"),
          getOrStop(conf, "user_consumed_path"),
          getOrStop(conf, "embed_size").toInt,
          getOrStop(conf, "learning_rate").toDouble,
          conf.getOrElse("iteration_number", "100").toInt,
          conf.getOrElse("show_progress_interval", "1").toInt,
          conf.getOrElse("topk_number", "10").toInt,
          conf.getOrElse("beam_size", "20").toInt,
          getOrStop(conf, "model_path"),
          getOrStop(conf, "embed_path")
        )
      case "cluster" =>
        ClusterTreeParams(
          getOrStop(conf, "embed_path"),
          conf.getOrElse("parallel", "true").toBoolean,
          Engine.coreNumber(),
          conf.getOrElse("cluster_num", "10").toInt,
          getOrStop(conf, "tree_protobuf_path"),
          conf.getOrElse("cluster_type", "kmeans")
        )
    }
  }

  def recommend(tdmModel: TDM): Unit = {
    val sequence = Array(0, 0, 2126, 204, 3257, 3439, 996, 1681, 3438, 1882)
    val rec = tdmModel.recommend(sequence, topk = 3, candidateNum = 20)
    println(s"Recommendation result: ${rec.mkString("Array(", ", ", ")")}")

    (1 to 10).foreach(_ => tdmModel.recommend(sequence, topk = 10, candidateNum = 20))
    val start = System.nanoTime()
    (1 to 100).foreach(_ => tdmModel.recommend(sequence, topk = 10, candidateNum = 20))
    val end = System.nanoTime()
    println(f"Average recommend time: ${(end - start) * 10 / 1e9d}%.4fms")
  }

  def showTime[T](block: => T, name: String): T = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    println(f"$name time: ${(t1 - t0) / 1e9d}%.4fs")
    result
  }
}
