package com.mass.retrieval.jtm

import cats.implicits._
import com.mass.retrieval.tdm.{getParameters, recommend, TrainModelParams}
import com.mass.scalann.nn.BCECriterionWithLogits
import com.mass.scalann.optim.Adam
import com.mass.scalann.utils.Property
import com.mass.tdm.dataset.LocalDataSet
import com.mass.tdm.model.{DeepFM, DIN, TDM}
import com.mass.tdm.optim.LocalOptimizer
import com.monovore.decline._

object JTMTrainDeepModel extends CommandApp(
  name = "TrainDeepModel",
  header = "JTM train deep model",
  main = {
    val fileOpt = Opts.option[String](
      long = "jtmConfFile",
      help = "JTM config file path, default path is `jtm.conf` from resource folder",
      metavar = "file"
    ).withDefault("fromResource")
    val quietOpt = Opts.flag("quiet", "Whether to be quiet.").orFalse

    (fileOpt, quietOpt).mapN { (jtmConfFile, quiet) =>
      val conf = Property.readConf(jtmConfFile, "model", "jtm", print = !quiet)
      Property.configLocal(conf)
      val params = getParameters(conf, "model") match {
        case p: TrainModelParams => p
        case _ => throw new IllegalArgumentException("wrong param type")
      }

      val dataset = LocalDataSet(
        trainPath = params.trainPath,
        evalPath = params.evalPath,
        pbFilePath = params.pbFilePath,
        userConsumedPath = params.userConsumedPath,
        totalTrainBatchSize = params.totalBatchSize,
        totalEvalBatchSize = params.totalEvalBatchSize,
        seqLen = params.seqLen,
        layerNegCounts = params.layerNegCounts,
        withProb = params.withProb,
        startSampleLevel = params.startSampleLevel,
        tolerance = params.tolerance,
        numThreads = params.numThreads,
        useMask = params.useMask
      )
      val deepModel =
        if (params.deepModelName == "din") {
          DIN.buildModel[Float](params.embedSize)
        } else if (params.deepModelName == "deepfm") {
          DeepFM.buildModel(params.seqLen, params.embedSize)
        } else {
          throw new IllegalArgumentException("DeepModel name should either be DeepFM or DIN")
        }

      val optimizer = LocalOptimizer(
        model = deepModel,
        dataset = dataset,
        criterion = BCECriterionWithLogits(),
        optimMethod = Adam(learningRate = params.learningRate),
        numIteration = params.numIteration,
        progressInterval = params.progressInterval,
        topk = params.topk,
        candidateNum = params.candidateNum,
        useMask = params.useMask
      )
      optimizer.optimize()

      val tdmModel = TDM(deepModel, params.deepModelName)
      TDM.saveModel(params.modelPath, params.embedPath, deepModel, params.embedSize)
      recommend(tdmModel)
    }
  }
)
