package com.mass.retrieval.jtm

import com.mass.jtm.optim.JTM
import com.mass.jtm.tree.TreeUtil
import com.mass.scalann.utils.{Engine, Property}
import com.mass.scalann.utils.Property.getOrStop
import com.mass.tdm.utils.Utils.time
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

object JTMTreeLearning {

  case class Params(jtmConfFile: String = "fromResource")

  def main(args: Array[String]): Unit = {
    Logger.getLogger("com.mass").setLevel(Level.INFO)

    val defaultParams = Params()
    val parser = new OptionParser[Params]("TreeLearning") {
      opt[String]("jtmConfFile")
        .text(s"JTM config file path, default path is `jtm.conf` from resource folder")
        .action((x, c) => c.copy(jtmConfFile = x))
    }

    parser.parse(args, defaultParams) match {
      case Some(params) => run(params)
      case _ => sys.exit(1)
    }
  }

  def run(params: Params): Unit = {
    val conf = Property.readConf(params.jtmConfFile, "tree", "jtm", print = true)
    Property.configLocal(conf)

    val dataPath = getOrStop(conf, "data_path")
    val treePath = getOrStop(conf, "tree_protobuf_path")
    val modelPath = getOrStop(conf, "model_path")
    val gap = getOrStop(conf, "gap").toInt
    val seqLen = getOrStop(conf, "seq_len").toInt
    val hierarchical = getOrStop(conf, "hierarchical_preference").toBoolean
    val minLevel = getOrStop(conf, "min_level").toInt
    val numThreads = Engine.coreNumber()
    val useMask = if (getOrStop(conf, "deep_model").toLowerCase == "din") true else false

    val jtm = JTM(
      dataPath = dataPath,
      treePath = treePath,
      modelPath = modelPath,
      gap = gap,
      seqLen = seqLen,
      hierarchical = hierarchical,
      minLevel = minLevel,
      numThreads = numThreads,
      useMask = useMask
    )
    val projectionPi = time(jtm.optimize(), s"JTM tree learning")
    TreeUtil.writeTree(jtm, projectionPi, treePath)
  }
}
