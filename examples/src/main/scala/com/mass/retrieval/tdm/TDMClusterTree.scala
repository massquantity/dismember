package com.mass.retrieval.tdm

import cats.implicits._
import com.mass.scalann.utils.{Engine, Property}
import com.mass.scalann.utils.Property.getOrStop
import com.mass.tdm.cluster.RecursiveCluster
import com.monovore.decline._

object TDMClusterTree extends CommandApp(
  name = "ClusterTree",
  header = "TDM using cluster to get a new tree",
  main = {
    val fileOpt = Opts.option[String](
      long = "tdmConfFile",
      help = "TDM config file path, default path is `tdm.conf` from resource folder",
      metavar = "file"
    ).withDefault("fromResource")
    val quietOpt = Opts.flag("quiet", "Whether to be quiet.").orFalse

    (fileOpt, quietOpt).mapN { (tdmConfFile, quiet) =>
      val conf = Property.readConf(tdmConfFile, "cluster", "tdm", print = !quiet)
      Property.configLocal(conf)
      val params = getParameters(conf, "cluster") match {
        case p: ClusterTreeParams => p
        case _ => throw new IllegalArgumentException("wrong param type")
      }

      val model = RecursiveCluster(
        embedPath = params.embedPath,
        parallel = params.parallel,
        numThreads = params.numThreads,
        clusterIterNum = params.clusterIterNum,
        clusterType = params.clusterType
      )
      showTime(model.run(params.outputTreePath), s"tree ${params.clusterType} clustering")
    }
  }
)
