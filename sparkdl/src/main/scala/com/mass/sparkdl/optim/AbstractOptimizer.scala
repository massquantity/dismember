package com.mass.sparkdl.optim

import scala.reflect.ClassTag

import org.apache.spark.rdd.RDD
import com.mass.sparkdl.Module
import com.mass.sparkdl.optim.DistriOptimizer.Cache
import com.mass.sparkdl.parameters.AllReduceParameter
import com.mass.sparkdl.tensor.TensorNumeric

abstract class AbstractOptimizer {

  protected def getModel[T: ClassTag](
      models: RDD[Cache[T]],
      parameters: AllReduceParameter[T],
      trainingModel: Module[T])(implicit ev: TensorNumeric[T]): Module[T]

  private[sparkdl] def clearState[T: ClassTag](models: RDD[DistriOptimizer.Cache[T]]): Unit = { }

  private[sparkdl] def endEpoch[T: ClassTag](optimMethods: Map[String, OptimMethod[T]]): Unit = {
    optimMethods.foreach { case (moduleName, optimMethod) =>
      val records = optimMethod.state.get[Int]("recordsProcessedThisEpoch")
      if (records.isDefined && records.get != 0) {
        optimMethod.state("epoch") = optimMethod.state[Int]("epoch") + 1
        optimMethod.state("recordsProcessedThisEpoch") = 0
      }
    }
  }

}
