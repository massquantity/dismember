package com.mass.retrieval.tdm

import com.mass.sparkdl.nn.BCECriterionWithLogits
import com.mass.sparkdl.optim.Adam
import com.mass.sparkdl.utils.{Engine, Property}
import com.mass.sparkdl.utils.Property.getOrStop
import com.mass.tdm.dataset.LocalDataSet
import com.mass.tdm.model.TDM
import com.mass.tdm.optim.LocalOptimizer
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

object TrainLocal {

  case class Params(tdmConfFile: String = "fromResource")

  def main(args: Array[String]): Unit = {
    Logger.getLogger("com.mass").setLevel(Level.INFO)

    val defaultParams = Params()
    val parser = new OptionParser[Params]("TrainLocal") {
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
    val conf = Property.readConf(path = params.tdmConfFile, prefix = "model", print = true)
    Property.configLocal(conf)

    val deepModel = getOrStop(conf, "deep_model").toLowerCase
    val seqLen = getOrStop(conf, "seq_len").toInt
    // val paddingId = 0  padded original item id
    val paddingIndex = conf.getOrElse("padding_index", "-1").toInt   // padded index or code
    val useMask = if (deepModel == "din") true else false

    val totalBatchSize = getOrStop(conf, "total_batch_size").toInt
    val totalEvalBatchSize = conf.getOrElse("total_eval_batch_size", "-1").toInt
    val evaluate = conf.getOrElse("evaluate_during_training", "false").toBoolean
    val layerNegCounts = getOrStop(conf, "layer_negative_counts")
    val withProb = conf.getOrElse("sample_with_probability", "true").toBoolean
    val startSampleLayer = conf.getOrElse("start_sample_layer", "-1").toInt
    val tolerance = conf.getOrElse("sample_tolerance", "20").toInt
    val numThreads = Engine.coreNumber()
    val parallelSample = conf.getOrElse("parallel_sample", "true").toBoolean

    val dataPath = getOrStop(conf, "train_path")
    val pbFilePath = getOrStop(conf, "tree_protobuf_path")
    val evalPath = conf.get("eval_path")

    val embedSize = getOrStop(conf, "embed_size").toInt
    val learningRate = getOrStop(conf, "learning_rate").toDouble
    val numIteration = conf.getOrElse("iteration_number", "100").toInt
    val progressInterval = conf.getOrElse("show_progress_interval", "1").toInt
    val topk = conf.getOrElse("topk_number", "10").toInt
    val candidateNum = conf.getOrElse("candidate_num_per_layer", "20").toInt

    val modelPath = getOrStop(conf, "model_path")
    val embedPath = getOrStop(conf, "embed_path")

    val dataset = new LocalDataSet(
      totalBatchSize = totalBatchSize,
      totalEvalBatchSize = totalEvalBatchSize,
      evaluate = evaluate,
      seqLen = seqLen,
      layerNegCounts = layerNegCounts,
      withProb = withProb,
      startSampleLayer = startSampleLayer,
      tolerance = tolerance,
      numThreads = numThreads,
      parallelSample = parallelSample,
      useMask = useMask)

    dataset.readFile(
      dataPath = dataPath,
      pbFilePath = pbFilePath,
      evalPath = evalPath)

    val tdmModel = TDM(
      featSeqLen = seqLen,
      embedSize = embedSize,
      deepModel = deepModel,
      paddingIndex = paddingIndex)

    val optimizer = new LocalOptimizer(
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
