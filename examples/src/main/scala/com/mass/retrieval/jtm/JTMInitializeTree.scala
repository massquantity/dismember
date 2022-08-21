package com.mass.retrieval.jtm

import cats.implicits._
import com.mass.retrieval.tdm.{getParameters, InitTreeParams}
import com.mass.scalann.utils.Property
import com.mass.tdm.tree.TreeInit
import com.monovore.decline._

object JTMInitializeTree extends CommandApp(
  name = "InitializeTree",
  header = "JTM tree initialization",
  main = {
    val fileOpt = Opts.option[String](
      long = "jtmConfFile",
      help = "JTM config file path, default path is `jtm.conf` from resource folder",
      metavar = "file"
    ).withDefault("fromResource")
    val quietOpt = Opts.flag("quiet", "Whether to be quiet.").orFalse

    (fileOpt, quietOpt).mapN { (jtmConfFile, quiet) =>
      val conf = Property.readConf(jtmConfFile, "init", "jtm", print = !quiet)
      val params = getParameters(conf, "init") match {
        case p: InitTreeParams => p
        case _ => throw new IllegalArgumentException("wrong param type")
      }

      val tree = new TreeInit(
        seqLen = params.seqLen,
        minSeqLen = params.minSeqLen,
        splitForEval = params.splitForEval,
        splitRatio = params.splitRatio
      )

      tree.generate(
        dataFile = params.dataPath,
        trainFile = params.trainPath,
        evalFile = params.evalPath,
        statFile = params.statPath,
        leafIdFile = params.leafIdPath,
        treePbFile = params.treePbPath,
        userConsumedFile = params.userConsumedPath
      )
    }
  }
)
