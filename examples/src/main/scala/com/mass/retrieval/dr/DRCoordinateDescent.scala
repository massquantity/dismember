package com.mass.retrieval.dr

import cats.implicits._
import com.mass.dr.dataset.LocalDataSet
import com.mass.dr.model.{DeepRetrieval, MappingOp}
import com.mass.dr.optim.CoordinateDescent
import com.mass.scalann.utils.Property
import com.monovore.decline._

object DRCoordinateDescent extends CommandApp(
  name = "CoordinateDescent",
  header = "Deep Retrieval M-step Coordinate Descent",
  main = {
    val fileOpt = Opts.option[String](
      long = "drConfFile",
      help = "Deep Retrieval config file path, default path is `deep-retrieval.conf` from resource folder",
      metavar = "file"
    ).withDefault("fromResource")
    val quietOpt = Opts.flag("quiet", "Whether to be quiet.").orFalse

    (fileOpt, quietOpt).mapN { (drConfFile, quiet) =>
      val conf = Property.readConf(drConfFile, "cd", "deep-retrieval", print = !quiet)
      Property.configLocal(conf)
      val params = getParameters(conf, "cd") match {
        case p: CoordinateDescentParams => p
        case _ => throw new IllegalArgumentException("wrong param type")
      }

      val dataset = LocalDataSet(
        numLayer = params.numLayer,
        numNode = params.numNode,
        numPathPerItem = params.numPathPerItem,
        trainBatchSize = params.trainBatchSize,
        evalBatchSize = params.evalBatchSize,
        seqLen = params.seqLen,
        minSeqLen = params.minSeqLen,
        dataPath = params.dataPath,
        mappingPath = params.mappingPath,
        initMapping = params.initMapping,
        splitRatio = params.splitRatio,
        delimiter = ","
      )

      val drModel = DeepRetrieval.loadModel(params.modelPath)
      val coo = CoordinateDescent(
        dataset = dataset,
        batchSize = params.trainBatchSize,
        numIteration = params.numIteration,
        numCandidatePath = params.numCandidatePath,
        numPathPerItem = params.numPathPerItem,
        numLayer = params.numLayer,
        numNode = params.numNode,
        decayFactor = params.decayFactor,
        penaltyFactor = params.penaltyFactor,
        penaltyPolyOrder = params.penaltyPolyOrder
      )
      val itemPathMapping = showTime(
        coo.optimize(drModel.layerModel, params.trainMode),
        "coordinate descent"
      )
      MappingOp.writeMapping(params.mappingPath, dataset.itemIdMapping, itemPathMapping)
    }
  }
)
