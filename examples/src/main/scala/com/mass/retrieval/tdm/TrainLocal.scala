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

    val dataset = new LocalDataSet(
      totalBatchSize = getOrStop(conf, "total_batch_size").toInt,
      totalEvalBatchSize = conf.getOrElse("total_eval_batch_size", "-1").toInt,
      evaluate = conf.getOrElse("evaluate_during_training", "false").toBoolean,
      seqLen = getOrStop(conf, "seq_len").toInt + 1,
      layerNegCounts = getOrStop(conf, "layer_negative_counts"),
      withProb = conf.getOrElse("sample_with_probability", "true").toBoolean,
      startSampleLayer = conf.getOrElse("start_sample_layer", "-1").toInt,
      tolerance = conf.getOrElse("sample_tolerance", "20").toInt,
      numThreads = Engine.coreNumber(),
      parallelSample = conf.getOrElse("parallel_sample", "true").toBoolean)

    dataset.readFile(
      dataPath = getOrStop(conf, "train_path"),
      pbFilePath = getOrStop(conf, "tree_protobuf_path"),
      evalPath = conf.get("eval_path"))

    val tdmModel = TDM(
      featSeqLen = conf("seq_len").toInt + 1,
      embedSize = getOrStop(conf, "embed_size").toInt)

    val optimizer = new LocalOptimizer(
      model = tdmModel.getModel,
      dataset = dataset,
      criterion = BCECriterionWithLogits(),
      optimMethod = Adam[Float](learningRate = getOrStop(conf, "learning_rate").toDouble),
      numIteration = conf.getOrElse("iteration_number", "100").toInt,
      progressInterval = conf.getOrElse("show_progress_interval", "1").toInt,
      topk = conf.getOrElse("topk_number", "10").toInt,
      candidateNum = conf.getOrElse("candidate_num_per_layer", "20").toInt)
    optimizer.optimize()

    TDM.saveModel(getOrStop(conf, "model_path"), getOrStop(conf, "embed_path"), tdmModel)

    recommend(tdmModel)
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
