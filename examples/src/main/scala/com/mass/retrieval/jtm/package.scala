package com.mass.retrieval

import com.mass.scalann.utils.Engine
import com.mass.scalann.utils.Property.getOrStop

package object jtm {

  sealed trait Params

  case class TreeLearningParams(
    trainPath: String,
    treePath: String,
    modelPath: String,
    gap: Int,
    seqLen: Int,
    hierarchical: Boolean,
    minLevel: Int,
    numThreads: Int,
    useMask: Boolean
  ) extends Params

  def getParameters(conf: Map[String, String], mode: String): Params = {
    mode match {
      case "tree" =>
        val useMask = if (getOrStop(conf, "deep_model").toLowerCase == "din") true else false
        TreeLearningParams(
          getOrStop(conf, "data_path"),
          getOrStop(conf, "tree_protobuf_path"),
          getOrStop(conf, "model_path"),
          getOrStop(conf, "gap").toInt,
          getOrStop(conf, "seq_len").toInt,
          getOrStop(conf, "hierarchical_preference").toBoolean,
          getOrStop(conf, "min_level").toInt,
          Engine.coreNumber(),
          useMask
        )
    }
  }
}
