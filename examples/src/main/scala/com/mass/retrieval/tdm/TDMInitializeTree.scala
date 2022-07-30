package com.mass.retrieval.tdm

import com.mass.scalann.utils.Property
import com.mass.scalann.utils.Property.getOrStop
import com.mass.tdm.tree.TreeInit
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

object TDMInitializeTree {

  case class Params(tdmConfFile: String = "fromResource")

  def main(args: Array[String]): Unit = {
    Logger.getLogger("com.mass").setLevel(Level.INFO)

    val defaultParams = Params()
    val parser = new OptionParser[Params]("InitializeTree") {
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
    val conf = Property.readConf(path = params.tdmConfFile, prefix = "init", print = true)

    val tree = new TreeInit(
      seqLen = getOrStop(conf, "seq_len").toInt,
      minSeqLen = getOrStop(conf, "min_seq_len").toInt,
      splitForEval = conf.getOrElse("split_for_eval", "false").toBoolean,
      splitRatio = conf.getOrElse("split_ratio", "0.8").toDouble
    )

    tree.generate(
      dataFile = getOrStop(conf, "data_path"),
      trainFile = getOrStop(conf, "train_path"),
      evalFile = conf.get("eval_path"),
      statFile = getOrStop(conf, "stat_path"),
      leafIdFile = getOrStop(conf, "leaf_id_path"),
      treePbFile = getOrStop(conf, "tree_protobuf_path"),
      userConsumedFile = conf.get("user_consumed_path")
    )
  }
}
