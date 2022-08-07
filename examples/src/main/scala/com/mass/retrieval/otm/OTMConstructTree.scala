package com.mass.retrieval.otm

import com.mass.otm.tree.TreeConstruction
import com.mass.scalann.utils.{Engine, Property}
import com.mass.scalann.utils.Property.getOrStop
import com.mass.tdm.utils.Serialization
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

object OTMConstructTree {

  case class Params(otmConfFile: String = "fromResource")

  def main(args: Array[String]): Unit = {
    Logger.getLogger("com.mass").setLevel(Level.INFO)

    val defaultParams = Params()
    val parser = new OptionParser[Params]("ConstructTree") {
      opt[String]("otmConfFile")
        .text(s"OTM config file path, default path is `otm.conf` from resource folder")
        .action((x, c) => c.copy(otmConfFile = x))
    }

    parser.parse(args, defaultParams) match {
      case Some(params) => run(params)
      case _ => sys.exit(1)
    }
  }

  def run(params: Params): Unit = {
    val conf = Property.readConf(params.otmConfFile, "tree", "otm", print = true)
    Property.configLocal(conf)

    val dataPath = getOrStop(conf, "data_path")
    val modelPath = getOrStop(conf, "model_path")
    val mappingPath = getOrStop(conf, "mapping_path")
    val deepModelName = getOrStop(conf, "deep_model").toLowerCase
    val useMask = if (deepModelName == "din") true else false
    val gap = getOrStop(conf, "gap").toInt
    val labelNum = getOrStop(conf, "label_num").toInt
    val seqLen = getOrStop(conf, "seq_len").toInt
    val minSeqLen = getOrStop(conf, "min_seq_len").toInt
    val splitRatio = getOrStop(conf, "split_ratio").toDouble
    val numThreads = Engine.coreNumber()

    val tree = TreeConstruction(
      dataPath = dataPath,
      modelPath = modelPath,
      mappingPath = mappingPath,
      gap = gap,
      labelNum = labelNum,
      minSeqLen = minSeqLen,
      seqLen = seqLen,
      splitRatio = splitRatio,
      numThreads = numThreads,
      useMask = useMask
    )
    val resultMapping = tree.run()
    Serialization.saveMapping(mappingPath, resultMapping)
  }
}
