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

    val dataset = new DistDataSet(
      totalBatchSize = getOrStop(modelConf, "total_batch_size").toInt,
      totalEvalBatchSize = modelConf.getOrElse("total_eval_batch_size", "-1").toInt,
      evaluate = modelConf.getOrElse("evaluate_during_training", "false").toBoolean,
      seqLen = getOrStop(modelConf, "seq_len").toInt + 1,
      layerNegCounts = getOrStop(modelConf, "layer_negative_counts"),
      withProb = modelConf.getOrElse("sample_with_probability", "true").toBoolean,
      startSampleLayer = modelConf.getOrElse("start_sample_layer", "-1").toInt,
      tolerance = modelConf.getOrElse("sample_tolerance", "20").toInt,
      numThreadsPerNode = Engine.coreNumber(),
      parallelSample = modelConf.getOrElse("parallel_sample", "true").toBoolean)

    dataset.readRDD(
      sc = sc,
      dataPath = getOrStop(modelConf, "train_path"),
      pbFilePath = getOrStop(modelConf, "tree_protobuf_path"),
      evalPath = modelConf.get("eval_path"))

    val tdmModel = TDM(
      featSeqLen = modelConf("seq_len").toInt + 1,
      embedSize = getOrStop(modelConf, "embed_size").toInt)

    val optimizer = new DistOptimizer(
      model = tdmModel.getModel,
      dataset = dataset,
      criterion = BCECriterionWithLogits(),
      optimMethod = Adam[Float](learningRate = getOrStop(modelConf, "learning_rate").toDouble),
      numIteration = modelConf.getOrElse("iteration_number", "100").toInt,
      progressInterval = modelConf.getOrElse("show_progress_interval", "1").toInt,
      topk = modelConf.getOrElse("topk_number", "10").toInt,
      candidateNum = modelConf.getOrElse("candidate_num_per_layer", "20").toInt)
    optimizer.optimize()

    TDM.saveModel(getOrStop(modelConf, "model_path"), getOrStop(modelConf, "embed_path"), tdmModel)

    recommend(tdmModel)

    sc.stop()
  }

  def recommend(tdmModel: TDM): Unit = {
    import com.mass.tdm.utils.Utils.time
    val sequence = Array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    time(println(tdmModel.recommend(sequence, 3).mkString("Array(", ", ", ")")), "recommend")
    time(println(tdmModel.recommend(sequence, 3).mkString("Array(", ", ", ")")), "recommend")
    time(println(tdmModel.recommend(sequence, 3).mkString("Array(", ", ", ")")), "recommend")
    time(println(tdmModel.recommend(sequence, 3).mkString("Array(", ", ", ")")), "recommend")
    time(println(tdmModel.recommend(sequence, 3).mkString("Array(", ", ", ")")), "recommend")
    time(println(tdmModel.recommend(sequence, 3).mkString("Array(", ", ", ")")), "recommend")
  }
}
