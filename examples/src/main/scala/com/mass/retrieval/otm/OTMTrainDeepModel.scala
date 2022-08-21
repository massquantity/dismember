package com.mass.retrieval.otm

import cats.implicits._
import com.mass.otm.dataset.LocalDataSet
import com.mass.otm.model.{DeepFM, DIN, OTM}
import com.mass.otm.optim.LocalOptimizer
import com.mass.scalann.utils.Property
import com.monovore.decline._

object OTMTrainDeepModel extends CommandApp(
  name = "TrainDeepModel",
  header = "OTM train deep model",
  main = {
    val fileOpt = Opts.option[String](
      long = "otmConfFile",
      help = "OTM config file path, default path is `otm.conf` from resource folder",
      metavar = "file"
    ).withDefault("fromResource")
    val quietOpt = Opts.flag("quiet", "Whether to be quiet.").orFalse

    (fileOpt, quietOpt).mapN { (otmConfFile, quiet) =>
      val conf = Property.readConf(otmConfFile, "model", "otm", print = !quiet)
      Property.configLocal(conf)
      val params = getParameters(conf, "model") match {
        case p: TrainModelParams => p
        case _ => throw new IllegalArgumentException("wrong param type")
      }
      require(params.trainBatchSize >= params.numThreads)

      val dataset = LocalDataSet(
        dataPath = params.dataPath,
        seqLen = params.seqLen,
        minSeqLen = params.minSeqLen,
        splitRatio = params.splitRatio,
        leafInitMode = params.leafInitMode,
        initMapping = params.initMapping,
        mappingPath = params.mappingPath,
        labelNum = params.labelNum,
        seed = params.seed,
      )
      val deepModel =
        if (params.deepModelName == "din") {
          DIN.buildModel[Double](params.embedSize, dataset.numTreeNode)
        } else if (params.deepModelName == "deepfm") {
          DeepFM.buildModel[Double](params.seqLen, params.embedSize, dataset.numTreeNode)
        } else {
          throw new IllegalArgumentException("DeepModel name should either be DeepFM or DIN")
        }

      val optimizer = LocalOptimizer(
        deepModel = deepModel,
        dataset = dataset,
        targetMode = params.targetMode,
        numEpoch = params.numEpoch,
        totalTrainBatchSize = params.trainBatchSize,
        totalEvalBatchSize = params.evalBatchSize,
        learningRate = params.learningRate,
        beamSize = params.beamSize,
        topk = params.topk,
        seqLen = params.seqLen,
        useMask = params.useMask,
        progressInterval = params.progressInterval
      )
      optimizer.optimize()

      OTM.saveModel(params.modelPath, params.mappingPath, deepModel, dataset.itemIdMapping)
      val otm = OTM(deepModel, dataset.itemIdMapping, params.deepModelName)
      recommend(otm)
    }
  }
)
