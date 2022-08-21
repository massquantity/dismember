package com.mass.retrieval.jtm

import cats.implicits._
import com.mass.jtm.optim.JTM
import com.mass.jtm.tree.TreeUtil
import com.mass.retrieval.tdm.showTime
import com.mass.scalann.utils.Property
import com.monovore.decline._

object JTMTreeLearning extends CommandApp(
  name = "TreeLearning",
  header = "JTM tree learning",
  main = {
    val fileOpt = Opts.option[String](
      long = "jtmConfFile",
      help = "JTM config file path, default path is `jtm.conf` from resource folder",
      metavar = "file"
    ).withDefault("fromResource")
    val quietOpt = Opts.flag("quiet", "Whether to be quiet.").orFalse

    (fileOpt, quietOpt).mapN { (jtmConfFile, quiet) =>
      val conf = Property.readConf(jtmConfFile, "tree", "jtm", print = !quiet)
      Property.configLocal(conf)
      val params = getParameters(conf, "tree") match {
        case p: TreeLearningParams => p
        case _ => throw new IllegalArgumentException("wrong param type")
      }

      val jtm = JTM(
        dataPath = params.trainPath,
        treePath = params.treePath,
        modelPath = params.modelPath,
        gap = params.gap,
        seqLen = params.seqLen,
        hierarchical = params.hierarchical,
        minLevel = params.minLevel,
        numThreads = params.numThreads,
        useMask = params.useMask
      )
      val projectionPi = showTime(jtm.optimize(), s"JTM tree learning")
      TreeUtil.writeTree(jtm, projectionPi, params.treePath)
    }
  }
)
