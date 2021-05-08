package com.mass.sparkdl.parameters

import java.nio.ByteBuffer
import java.util.concurrent._
import java.util.concurrent.atomic.AtomicLong

import scala.collection.JavaConverters._
import scala.reflect.{classTag, ClassTag}

import com.mass.sparkdl.tensor.{Tensor, TensorNumeric}
import com.mass.sparkdl.utils.Engine
import org.apache.commons.lang.exception.ExceptionUtils
import org.apache.log4j.Logger
import org.apache.spark.TaskContext
import org.apache.spark.sparkExtension.SparkExtension
import org.apache.spark.storage.{BlockId, BlockManagerWrapper, StorageLevel}

object AllReduceParameter {

  val logger: Logger = Logger.getLogger(getClass)
  private val syncPoolSize: Int = 4
  private val computePoolSize: Int = Math.max(Runtime.getRuntime.availableProcessors() / 2, 1)

  val syncPool: ExecutorService = Executors.newFixedThreadPool(syncPoolSize, new ThreadFactory {
    override def newThread(r: Runnable): Thread = {
      val t = Executors.defaultThreadFactory().newThread(r)
      t.setDaemon(true)
      t
    }
  })

  val computePool: ExecutorService = Executors.newFixedThreadPool(computePoolSize,
    (r: Runnable) => {
      val t = Executors.defaultThreadFactory().newThread(r)
      t.setDaemon(true)
      t
    }
  )

  private val nextId = new AtomicLong(0)

  def newParameter[T: ClassTag](
      partitionNum: Int,
      size: Int,
      offset: Int = 0,
      compress: String = "fp16")(implicit ev: TensorNumeric[T]): AllReduceParameter[T] = {
    new AllReduceParameter(nextId.getAndIncrement(), partitionNum, size, offset, compress)
  }
}

class AllReduceParameter[T: ClassTag](id: Long, partitionNum: Int, val size: Int,
    val paramOffset: Int = 0, val compress: String = "fp16")(implicit ev: TensorNumeric[T])
    extends Serializable {
  import AllReduceParameter._

  @transient private var taskSize = 0
  @transient private var extraSize = 0
  @transient private var partitionId: Int = 0
  @transient lazy val weightPartition: Tensor[T] = readWeightPartition()
  @transient lazy val gradientPartition: Tensor[T] = readGradientPartition()

  private def readObject(in: java.io.ObjectInputStream): Unit = {
    in.defaultReadObject()
    taskSize = size / partitionNum
    extraSize = size % partitionNum
    partitionId = TaskContext.getPartitionId()
  }

  private def getWeightBlockId(pid: Int): BlockId = {
    SparkExtension.getLocalBlockId(id + "weightBytes" + pid)
  }

  private def getWeightPartitionId(): BlockId = {
    SparkExtension.getLocalBlockId(id + "weights" + partitionId)
  }

  private def getGradientPartitionId(): BlockId = {
    SparkExtension.getLocalBlockId(id + "gradients" + partitionId)
  }

  private def getGradientBlockId(pidFrom: Int, pidTo: Int): BlockId = {
    SparkExtension.getLocalBlockId(id.toString + "_" + pidTo + "gradientBytes" + pidFrom)
  }

  private def readWeightPartition(): Tensor[T] = {
    val blockId: BlockId = getWeightPartitionId()
    BlockManagerWrapper.getLocal(blockId)
      .map(_.data.next().asInstanceOf[Tensor[T]])
      .getOrElse(throw new IllegalStateException("Please initialize AllReduceParameter first!"))
  }

  private def readGradientPartition(): Tensor[T] = {
    val blockId = getGradientPartitionId()
    BlockManagerWrapper.getLocal(blockId)
      .map(_.data.next().asInstanceOf[Tensor[T]])
      .getOrElse(throw new IllegalStateException("Please initialize AllReduceParameter first!"))
  }

  def localPartitionRange: (Int, Int) = {
    (paramOffset + partitionId * taskSize + math.min(partitionId, extraSize),
      taskSize + (if (partitionId < extraSize) 1 else 0))
  }

  def init(parameter: Tensor[T])(implicit ev: TensorNumeric[T]): (Int, Int, Int) = {
    val _classTag = classTag[T]
    val start = partitionId * taskSize + math.min(partitionId, extraSize)
    val length = taskSize + (if (partitionId < extraSize) 1 else 0)

    val _weights = Tensor[T](length)(_classTag, ev).copy(parameter.narrow(0, start, length))
    val _gradients = Tensor[T](length)(_classTag, ev)

    BlockManagerWrapper.removeBlock(getWeightPartitionId())
    BlockManagerWrapper.putSingle(getWeightPartitionId(), _weights,
      StorageLevel.MEMORY_AND_DISK, tellMaster = false)
    BlockManagerWrapper.removeBlock(getGradientPartitionId())
    BlockManagerWrapper.putSingle(getGradientPartitionId(), _gradients,
      StorageLevel.MEMORY_AND_DISK, tellMaster = false)
    val blockId = getWeightBlockId(partitionId)
    val compressed: CompressedTensor[T] = SerializerInstance.create(length, compress)
    compressed.compress(0, parameter, start, length)
    BlockManagerWrapper.putBytes(blockId, compressed.bytes(), StorageLevel.MEMORY_ONLY_SER)
    (partitionId, start, length)
  }

  def getWeights(localParameter: Tensor[T]): FutureResult[Int] = {
    val tasks = (0 until partitionNum).map { pid =>
      syncPool.submit {
        new Callable[Int] {
          override def call(): Int = {
            try {
              val blockId = getWeightBlockId(pid)
              val localBuffer: ByteBuffer = BlockManagerWrapper.getLocalOrRemoteBytes(blockId).get
              val start = pid * taskSize + math.min(pid, extraSize)
              val length = taskSize + (if (pid < extraSize) 1 else 0)
              SerializerInstance.create(localBuffer, compress)
                .deCompress(0, localParameter, start, length)
              BlockManagerWrapper.unlock(blockId)
              pid
            } catch {
              case t: Throwable =>
                logger.error("Error: " + ExceptionUtils.getStackTrace(t))
                throw t
            }
          }
        }
      }
    }
    new FutureResult(tasks)
  }

  def putGradients(parameter: Tensor[T]): Unit = {
    computePool.invokeAll((0 until partitionNum).map(i =>
      new Callable[Int] {
        override def call(): Int = {
          val start = i * taskSize + math.min(i, extraSize)
          val length = taskSize + (if (i < extraSize) 1 else 0)
          val blockId = getGradientBlockId(partitionId, i)
          val block = BlockManagerWrapper.getLocalBytes(blockId)
          if (block.isDefined) {
            val compressed: CompressedTensor[T] = SerializerInstance.create(block.get, compress)
            compressed.compress(0, parameter, start, length)
            i
          } else {
            val compressed: CompressedTensor[T] = SerializerInstance.create(length, compress)
            compressed.compress(0, parameter, start, length)
            BlockManagerWrapper.putBytes(blockId, compressed.bytes(), StorageLevel.MEMORY_ONLY_SER)
            i
          }
        }
      }
    ).asJava)
  }

  def aggregateGradientPartition(avgNumbers: Int): Unit = {
    require(partitionId < partitionNum, s"This parameter was created with $partitionNum " +
      s"partitions. It cannot be used on RDDs with > $partitionNum partitions.")
    val params = new Array[CompressedTensor[T]](partitionNum)
    val sgThreads = (0 until partitionNum).map { pid =>
      new Callable[Int] {
        override def call(): Int = {
          try {
            val blockId = getGradientBlockId(pid, partitionId)
            val tmp: ByteBuffer = BlockManagerWrapper.getLocalOrRemoteBytes(blockId).get
            params(pid) = SerializerInstance.create(tmp, compress)
            BlockManagerWrapper.unlock(blockId)
            pid
          } catch {
            case t: Throwable =>
              logger.error("Error: " + ExceptionUtils.getStackTrace(t))
              throw t
          }
        }
      }
    }
    syncPool.invokeAll(sgThreads.asJava)

    val length = taskSize + (if (partitionId < extraSize) 1 else 0)
    val poolSize = Engine.default.getPoolSize
    val innerTaskSize = length / poolSize
    val innerExtraSize = length % poolSize
    val availableTask = if (innerTaskSize == 0) innerExtraSize else poolSize
    computePool.invokeAll((0 until availableTask).map(tid =>
      new Callable[Int] {
        override def call(): Int = {
          val innerStart = tid * innerTaskSize + math.min(innerExtraSize, tid)
          val innerLength = innerTaskSize + (if (tid < innerExtraSize) 1 else 0)
          params.reduce { (l, r) =>
            l.add(r.bytes(innerStart, innerLength), innerStart, innerLength)
          }
          tid
        }
      }
    ).asJava)
    params.head.deCompress(gradientPartition)
    gradientPartition.div(ev.fromType(avgNumbers))
  }

  def sendWeightPartition(): Unit = {
    val blockId = getWeightBlockId(partitionId)
    val localBuffer = BlockManagerWrapper.getLocalBytes(blockId).get
    SerializerInstance.create(localBuffer, compress).compress(weightPartition)

    val weightsId = getWeightPartitionId()
    val weights = BlockManagerWrapper.getLocal(weightsId)
      .map(_.data.next().asInstanceOf[Tensor[T]])
      .get
    weights.copy(weightPartition)
  }
}

private[sparkdl] class FutureResult[T](private val futures: Seq[Future[T]]) {
  def waitResult(): Seq[T] = futures.map(_.get())
}
