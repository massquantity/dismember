package com.mass.retrieval.tdm

import com.mass.sparkdl.nn.BCECriterionWithLogits
import com.mass.sparkdl.optim.Adam
import com.mass.sparkdl.utils.{Engine, Property}
import com.mass.tdm.dataset.DistDataSet
import com.mass.tdm.model.TDM
import com.mass.tdm.optim.DistOptimizer
import Property.{getOption, getOrStop}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import scopt.OptionParser

object TrainDist {

  case class Params(tdmConfFile: String = "fromResource")

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("com").setLevel(Level.ERROR)
    Logger.getLogger("com.mass").setLevel(Level.INFO)

    val defaultParams = Params()
    val parser = new OptionParser[Params]("TrainDist") {
      opt[String]("tdmConfFile")
        .text(s"TDM config file path, default path is tdm.conf from resource folder")
        .action((x, c) => c.copy(tdmConfFile = x))
    }

    parser.parse(args, defaultParams) match {
      case Some(params) => run(params)
      case _ => sys.exit(1)
    }
  }

  def run(params: Params): Unit = {
    val modelConf = Property.readConf(path = params.tdmConfFile,
      prefix = "model", print = true)
    val sparkConf = Property.readConf(path = params.tdmConfFile,
      prefix = "spark", truncate = false, print = true)

    val sc: SparkContext = Property.configDist(sparkConf)

    val deepModel = getOrStop(modelConf, "deep_model").toLowerCase
    val seqLen = getOrStop(modelConf, "seq_len").toInt
    // val paddingId = 0  padded original item id
    val paddingIndex = modelConf.getOrElse("padding_index", "-1").toInt   // padded index or code
    val useMask = if (deepModel == "din") true else false

    val totalBatchSize = getOrStop(modelConf, "total_batch_size").toInt
    val totalEvalBatchSize = modelConf.getOrElse("total_eval_batch_size", "-1").toInt
    val evaluate = modelConf.getOrElse("evaluate_during_training", "false").toBoolean
    val layerNegCounts = getOrStop(modelConf, "layer_negative_counts")
    val withProb = modelConf.getOrElse("sample_with_probability", "true").toBoolean
    val startSampleLayer = modelConf.getOrElse("start_sample_layer", "-1").toInt
    val tolerance = modelConf.getOrElse("sample_tolerance", "20").toInt
    val numThreadsPerNode = Engine.coreNumber()
    val parallelSample = modelConf.getOrElse("parallel_sample", "true").toBoolean

    val dataPath = getOrStop(modelConf, "train_path")
    val pbFilePath = getOrStop(modelConf, "tree_protobuf_path")
    val evalPath = modelConf.get("eval_path")
    val userConsumedPath = modelConf.get("user_consumed_path")

    val embedSize = getOrStop(modelConf, "embed_size").toInt
    val learningRate = getOrStop(modelConf, "learning_rate").toDouble
    val numIteration = modelConf.getOrElse("iteration_number", "100").toInt
    val progressInterval = modelConf.getOrElse("show_progress_interval", "1").toInt
    val topk = modelConf.getOrElse("topk_number", "10").toInt
    val candidateNum = modelConf.getOrElse("candidate_num_per_layer", "20").toInt

    val modelPath = getOrStop(modelConf, "model_path")
    val embedPath = getOrStop(modelConf, "embed_path")

    val dataset = new DistDataSet(
      totalBatchSize = totalBatchSize,
      totalEvalBatchSize = totalEvalBatchSize,
      evaluate = evaluate,
      seqLen = seqLen,
      layerNegCounts = layerNegCounts,
      withProb = withProb,
      startSampleLevel = startSampleLayer,
      tolerance = tolerance,
      numThreadsPerNode = numThreadsPerNode,
      parallelSample = parallelSample,
      useMask = useMask)

    dataset.readRDD(
      sc = sc,
      dataPath = dataPath,
      pbFilePath = pbFilePath,
      evalPath = evalPath,
      userConsumedPath = userConsumedPath)

    val tdmModel = TDM(
      featSeqLen = seqLen,
      embedSize = embedSize,
      deepModel = deepModel,
      paddingIndex = paddingIndex)

    val optimizer = new DistOptimizer(
      model = tdmModel.getModel,
      dataset = dataset,
      criterion = BCECriterionWithLogits(),
      optimMethod = Adam[Float](learningRate = learningRate),
      numIteration = numIteration,
      progressInterval = progressInterval,
      topk = topk,
      candidateNum = candidateNum,
      useMask = useMask)

    optimizer.optimize()

    TDM.saveModel(modelPath, embedPath, tdmModel)

    recommend(tdmModel)

    sc.stop()
  }

  def recommend(tdmModel: TDM): Unit = {
    val sequence = Array(0, 0, 2126, 204, 3257, 3439, 996, 1681, 3438, 1882)
    println("Recommendation result: " +
      tdmModel.recommend(sequence, 3).mkString("Array(", ", ", ")"))

    (1 to 10).foreach(_ => tdmModel.recommend(sequence, topk = 10, candidateNum = 20))

    val start = System.nanoTime()
    var i = 0
    while (i < 100) {
      tdmModel.recommend(sequence, topk = 10, candidateNum = 20)
      i += 1
    }
    val end = System.nanoTime()
    println(f"Average recommend time: ${(end - start) * 10 / 1e9d}%.4fms")
  }
}
