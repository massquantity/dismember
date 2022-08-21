package com.mass.retrieval.otm

import cats.implicits._
import com.mass.otm.tree.TreeConstruction
import com.mass.retrieval.tdm.showTime
import com.mass.scalann.utils.Property
import com.mass.tdm.utils.Serialization
import com.monovore.decline._

object OTMConstructTree extends CommandApp(
  name = "ConstructTree",
  header = "OTM tree construction",
  main = {
    val fileOpt = Opts.option[String](
      long = "otmConfFile",
      help = "OTM config file path, default path is `otm.conf` from resource folder",
      metavar = "file"
    ).withDefault("fromResource")
    val quietOpt = Opts.flag("quiet", "Whether to be quiet.").orFalse

    (fileOpt, quietOpt).mapN { (otmConfFile, quiet) =>
      val conf = Property.readConf(otmConfFile, "tree", "otm", print = !quiet)
      Property.configLocal(conf)
      val params = getParameters(conf, "tree") match {
        case p: TreeConstructionParams => p
        case _ => throw new IllegalArgumentException("wrong param type")
      }

      val tree = TreeConstruction(
        dataPath = params.dataPath,
        modelPath = params.modelPath,
        mappingPath = params.mappingPath,
        gap = params.gap,
        labelNum = params.labelNum,
        minSeqLen = params.minSeqLen,
        seqLen = params.seqLen,
        splitRatio = params.splitRatio,
        numThreads = params.numThreads,
        useMask = params.useMask
      )
      val resultMapping= showTime(tree.run(), s"OTM tree construction")
      Serialization.saveMapping(params.mappingPath, resultMapping)
    }
  }
)
