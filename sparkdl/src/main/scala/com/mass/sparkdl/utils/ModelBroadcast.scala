package com.mass.sparkdl.utils

import java.io.{IOException, ObjectInputStream, ObjectOutputStream}
import java.util.UUID

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

import com.mass.sparkdl.Module
import com.mass.sparkdl.tensor.{Storage, Tensor, TensorNumeric}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

trait ModelBroadcast[T] extends Serializable {

  private val _uuid: String = UUID.randomUUID().toString

  def broadcast(sc: SparkContext, model: Module[T]): this.type

  def value(initGradient: Boolean, shareWeight: Boolean): Module[T]

  def uuid: String = _uuid
}

object ModelBroadcast {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): ModelBroadcast[T] = {
    if (System.getProperty("ModelBroadcastFactory") != null) {
      val cls = Class.forName(System.getProperty("ModelBroadcastFactory"))
      cls.getConstructors()(0).newInstance().asInstanceOf[ModelBroadcastFactory].create()
    } else {
      new DefaultModelBroadcastFactory().create()
    }
  }
}

private[sparkdl] class ModelBroadcastImp[T: ClassTag](applyProtoBuffer: Boolean = false)(
    implicit ev: TensorNumeric[T]) extends ModelBroadcast[T] {

  private var broadcastModel: Broadcast[ModelInfo[T]] = _
  private var broadcastParameters: Broadcast[Array[Tensor[T]]] = _
  private var nodeNumber : Int = _
  private var coreNumber : Int = _

  private def setNodeAndCore(): Unit = {
    nodeNumber = Engine.nodeNumber()
    coreNumber = Engine.coreNumber()
  }

  override def broadcast(sc: SparkContext, model: Module[T]): this.type = {
    CachedModels.deleteAll(uuid)

    if (applyProtoBuffer) {
      broadcastModel = sc.broadcast(ModelInfo(uuid, model))
    } else {
      // first clear parameters, then broadcast the light-weight model
      val weightBias = Util.getAndClearWeightBias(model.parameters())
      broadcastModel = sc.broadcast(ModelInfo[T](uuid, model))
      broadcastParameters = sc.broadcast(weightBias)

      Util.putWeightBias(Util.cloneParameters(weightBias), model)
      Util.initGradWeightBias(weightBias, model)
    }
    setNodeAndCore()
    this
  }

  override def value(initGradient: Boolean, shareWeight: Boolean): Module[T] = {
    Engine.setNodeAndCore(nodeNumber, coreNumber)
    CachedModels.deleteAll(uuid)

    if (applyProtoBuffer) {
      // todo
      /*
      val localModel = broadcastModel.value.model.clone(false)
      val uuid = broadcastModel.value.uuid
      CachedModels.add(uuid, localModel)
      if (initGradient) {
        Util.initGradWeightBias(getWeightBias(localModel.parameters()), localModel)
      }
      localModel
      */
      broadcastModel.value.model
    } else {
      val localModel = broadcastModel.value.model.cloneModule()
      val uuid = broadcastModel.value.uuid
      CachedModels.add(uuid, localModel)
      val parameters = {
        if (shareWeight) {
          broadcastParameters.value
        } else {
          Util.cloneParameters(broadcastParameters.value)
        }
      }

      Util.putWeightBias(parameters, localModel)
      if (initGradient) {
        Util.initGradWeightBias(broadcastParameters.value, localModel)
      }
      localModel
    }
  }

  def getWeightBias(parameters: (Array[Tensor[T]], Array[Tensor[T]])): Array[Tensor[T]] = {
    if (parameters._1.nonEmpty) {
      val weightBias = new Array[Tensor[T]](parameters._1.length)
      val first = parameters._1(0)
      val isCompacted = parameters._1.map(_.nElement()).sum == first.storage().length

      parameters._1.zipWithIndex.foreach { case (wb, i) =>
        if (wb != null) {
          weightBias(i) = {
            if (isCompacted) {
              Tensor[T](Storage(first.storage().array()), wb.storageOffset(), wb.size(), wb.stride())
            } else {
              Tensor[T](Storage(wb.storage().array()), wb.storageOffset(), wb.size(), wb.stride())
            }
          }
        }
      }
      weightBias
    } else {
      Array.empty
    }
  }
}

private[sparkdl] class ModelInfo[T: ClassTag](val uuid: String, @transient var model: Module[T])(
    implicit ev: TensorNumeric[T]) extends Serializable {

  @throws(classOf[IOException])
  private def writeObject(out: ObjectOutputStream): Unit = {
    out.defaultWriteObject()
    val cloned = model.cloneModule()
    out.writeObject(cloned)
    CachedModels.add(uuid, cloned)
  }

  @throws(classOf[IOException])
  private def readObject(in: ObjectInputStream): Unit = {
    in.defaultReadObject()
    model = in.readObject().asInstanceOf[Module[T]]
    CachedModels.add(uuid, model)
  }
}

object ModelInfo {
  def apply[T: ClassTag](uuid: String, model: Module[T])(
    implicit ev: TensorNumeric[T]): ModelInfo[T] = new ModelInfo[T](uuid, model)
}

object CachedModels {
  import java.util.concurrent.ConcurrentHashMap

  import scala.collection.concurrent.{Map => ConcurrentMap}
  import collection.JavaConverters._

  private val cachedModels: ConcurrentMap[String, ArrayBuffer[Module[_]]] =
    new ConcurrentHashMap[String, ArrayBuffer[Module[_]]]().asScala

  def add[T: ClassTag](uuid: String, model: Module[T])(implicit ev: TensorNumeric[T]): Unit = {
    CachedModels.synchronized {
      val models = cachedModels.get(uuid) match {
        case Some(values) => values += model
        case _ => ArrayBuffer(model)
      }
      cachedModels.put(uuid, models.asInstanceOf[ArrayBuffer[Module[_]]])
    }
  }

  def deleteAll[T: ClassTag](currentKey: String)(implicit ev: TensorNumeric[T]): Unit = {
    CachedModels.synchronized {
      val keys = cachedModels.keys
      for (key <- keys) {
        if (key != currentKey) {
          val models = cachedModels(key)
          for (model <- models) {
            model.release()
          }
          cachedModels.remove(key)
        }
      }
    }
  }

  def deleteKey[T: ClassTag](key: String)(implicit ev: TensorNumeric[T]): Unit = {
    CachedModels.synchronized {
      val keys = cachedModels.keys
      for (k <- keys) {
        if (k == key) {
          val models = cachedModels(key)
          for (model <- models) {
            model.release()
          }
          cachedModels.remove(key)
        }
      }
    }
  }
}
