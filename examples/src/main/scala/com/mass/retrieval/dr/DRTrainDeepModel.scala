package com.mass.retrieval.dr

import cats.implicits._
import com.mass.dr.dataset.LocalDataSet
import com.mass.dr.model.{DeepRetrieval, LayerModel, MappingOp, RerankModel}
import com.mass.dr.optim.LocalOptimizer
import com.mass.scalann.utils.Property
import com.monovore.decline._

object DRTrainDeepModel extends CommandApp(
  name = "TrainDeepModel",
  header = "Deep Retrieval E-step Train Model",
  main = {
    val fileOpt = Opts.option[String](
      long = "drConfFile",
      help = "Deep Retrieval config file path, default path is `deep-retrieval.conf` from resource folder",
      metavar = "file"
    ).withDefault("fromResource")
    val quietOpt = Opts.flag("quiet", "Whether to be quiet.").orFalse

    (fileOpt, quietOpt).mapN { (drConfFile, quiet) =>
      val conf = Property.readConf(drConfFile, "model", "deep-retrieval", print = !quiet)
      Property.configLocal(conf)
      val params = getParameters(conf, "model") match {
        case p: TrainModelParams => p
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

      val layerModel = LayerModel(
        dataset.numItem,
        params.numNode,
        params.numLayer,
        params.seqLen,
        params.embedSize
      )
      val reRankModel = RerankModel(
        dataset.numItem,
        params.seqLen,
        params.embedSize
      )
      val optimizer = LocalOptimizer(
        dataset = dataset,
        layerModel = layerModel,
        reRankModel = reRankModel,
        numEpoch = params.numEpoch,
        numLayer = params.numLayer,
        learningRate = params.learningRate,
        numSampled = params.numSampled,
        embedSize = params.embedSize,
        topk = params.topk,
        beamSize = params.beamSize,
        progressInterval = params.progressInterval,
        reRankEpoch = None
      )
      optimizer.optimize()

      val mappingOp = MappingOp(dataset.itemIdMapping, dataset.itemPathMapping)
      val drModel = DeepRetrieval(
        layerModel = layerModel,
        reRankModel = reRankModel,
        numItem = dataset.numItem,
        numNode = params.numNode,
        numLayer = params.numLayer,
        seqLen = params.seqLen,
        embedSize = params.embedSize
      )
      DeepRetrieval.saveModel(drModel, params.modelPath)
      recommend(drModel, mappingOp)
    }
  }
)
