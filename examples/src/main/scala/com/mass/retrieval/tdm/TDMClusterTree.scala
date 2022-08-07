package com.mass.retrieval.tdm

import com.mass.scalann.utils.{Engine, Property}
import com.mass.scalann.utils.Property.getOrStop
import com.mass.tdm.cluster.RecursiveCluster
import com.mass.tdm.utils.Utils.time
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser

object TDMClusterTree {

  case class Params(tdmConfFile: String = "fromResource")

  def main(args: Array[String]): Unit = {
    Logger.getLogger("com.mass").setLevel(Level.INFO)

    val defaultParams = Params()
    val parser = new OptionParser[Params]("ClusterTree") {
      opt[String]("tdmConfFile")
        .text(s"TDM config file path, default path is `tdm.conf` from resource folder")
        .action((x, c) => c.copy(tdmConfFile = x))
    }

    parser.parse(args, defaultParams) match {
      case Some(params) => run(params)
      case _ => sys.exit(1)
    }
  }

  def run(params: Params): Unit = {
    val conf = Property.readConf(params.tdmConfFile, "cluster", "tdm", print = true)
    Property.configLocal(conf)

    val embedPath = getOrStop(conf, "embed_path")
    val parallel = conf.getOrElse("parallel", "true").toBoolean
    val numThreads = Engine.coreNumber()
    val clusterIterNum = conf.getOrElse("cluster_num", "10").toInt
    val outputTreePath = getOrStop(conf, "tree_protobuf_path")
    val clusterType = conf.getOrElse("cluster_type", "kmeans")

    val model = RecursiveCluster(
      embedPath = embedPath,
      parallel = parallel,
      numThreads = numThreads,
      clusterIterNum = clusterIterNum,
      clusterType = clusterType
    )
    time(model.run(outputTreePath), s"tree $clusterType clustering")
  }
}
