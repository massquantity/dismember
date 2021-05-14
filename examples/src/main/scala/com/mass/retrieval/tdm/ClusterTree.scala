package com.mass.retrieval.tdm

import java.net.URI

import com.mass.sparkdl.utils.{Engine, Property}
import com.mass.tdm.cluster.RecursiveCluster
import Property.getOrStop
import com.mass.tdm.utils.Utils.time
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

object ClusterTree {

  case class Params(tdmConfFile: String = "fromResource")

  def main(args: Array[String]): Unit = {
    Logger.getLogger("com.mass").setLevel(Level.INFO)

    val defaultParams = Params()
    val parser = new OptionParser[Params]("ClusterTree") {
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
    val conf = Property.readConf(path = params.tdmConfFile, prefix = "cluster", print = true)
    Property.configLocal(conf)

    val model = new RecursiveCluster(
      parallel = conf.getOrElse("parallel", "true").toBoolean,
      numThreads = Engine.coreNumber())

    time(model.run(getOrStop(conf, "embed_path"), getOrStop(conf, "tree_protobuf_path")), "cluster")

  }
}
