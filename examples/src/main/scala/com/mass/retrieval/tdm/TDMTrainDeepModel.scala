package com.mass.retrieval.tdm

import com.mass.scalann.nn.BCECriterionWithLogits
import com.mass.scalann.optim.Adam
import com.mass.scalann.utils.{Engine, Property}
import com.mass.scalann.utils.Property.getOrStop
import com.mass.tdm.dataset.LocalDataSet
import com.mass.tdm.model.TDM
import com.mass.tdm.optim.LocalOptimizer
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

object TDMTrainDeepModel {

  case class Params(tdmConfFile: String = "fromResource")

  def main(args: Array[String]): Unit = {
    Logger.getLogger("com.mass").setLevel(Level.INFO)

    val defaultParams = Params()
    val parser = new OptionParser[Params]("TrainDeepModel") {
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
    val useMask = if (deepModel == "din") true else false

    val totalBatchSize = getOrStop(conf, "total_batch_size").toInt
    val totalEvalBatchSize = getOrStop(conf, "total_eval_batch_size").toInt
    val layerNegCounts = getOrStop(conf, "layer_negative_counts")
    val withProb = conf.getOrElse("sample_with_probability", "true").toBoolean
    val startSampleLevel = conf.getOrElse("start_sample_level", "1").toInt
    val tolerance = conf.getOrElse("sample_tolerance", "20").toInt
    val numThreads = Engine.coreNumber()

    val dataPath = getOrStop(conf, "train_path")
    val pbFilePath = getOrStop(conf, "tree_protobuf_path")
    val evalPath = getOrStop(conf, "eval_path")
    val userConsumedPath = getOrStop(conf, "user_consumed_path")

    val embedSize = getOrStop(conf, "embed_size").toInt
    val learningRate = getOrStop(conf, "learning_rate").toDouble
    val numIteration = conf.getOrElse("iteration_number", "100").toInt
    val progressInterval = conf.getOrElse("show_progress_interval", "1").toInt
    val topk = conf.getOrElse("topk_number", "10").toInt
    val candidateNum = conf.getOrElse("candidate_num_per_layer", "20").toInt

    val modelPath = getOrStop(conf, "model_path")
    val embedPath = getOrStop(conf, "embed_path")

    val dataset = LocalDataSet(
      trainPath = dataPath,
      evalPath = evalPath,
      pbFilePath = pbFilePath,
      userConsumedPath = userConsumedPath,
      totalTrainBatchSize = totalBatchSize,
      totalEvalBatchSize = totalEvalBatchSize,
      seqLen = seqLen,
      layerNegCounts = layerNegCounts,
      withProb = withProb,
      startSampleLevel = startSampleLevel,
      tolerance = tolerance,
      numThreads = numThreads,
      useMask = useMask)

    val tdmModel = TDM(
      featSeqLen = seqLen,
      embedSize = embedSize,
      deepModel = deepModel)

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
    val rec = tdmModel.recommend(sequence, topk = 3, candidateNum = 20)
    println(s"Recommendation result: ${rec.mkString("Array(", ", ", ")")}")

    (1 to 10).foreach(_ => tdmModel.recommend(sequence, topk = 10, candidateNum = 20))
    val start = System.nanoTime()
    (1 to 100).foreach(_ => tdmModel.recommend(sequence, topk = 10, candidateNum = 20))
    val end = System.nanoTime()
    println(f"Average recommend time: ${(end - start) * 10 / 1e9d}%.4fms")
  }
}
