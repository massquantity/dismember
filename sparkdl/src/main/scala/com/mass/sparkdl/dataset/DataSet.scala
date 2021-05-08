package com.mass.sparkdl.dataset

import java.util.concurrent.atomic.AtomicInteger

import scala.reflect.ClassTag

import com.mass.sparkdl.DataSet
import com.mass.sparkdl.utils.Engine
import org.apache.commons.math3.random.RandomDataGenerator
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext

trait AbstractDataSet[D, DataSequence] {

  def data(train: Boolean): DataSequence

  def shuffle(): Unit

  def size(): Long

  def transform[C: ClassTag](transformer: Transformer[D, C]): DataSet[C]

  def -> [C: ClassTag](transformer: Transformer[D, C]): DataSet[C] = this.transform(transformer)

  def toLocal: LocalDataSet[D] = this.asInstanceOf[LocalDataSet[D]]

  def toDistributed: DistributedDataSet[D] = this.asInstanceOf[DistributedDataSet[D]]
}

trait LocalDataSet[T] extends AbstractDataSet[T, Iterator[T]] { self =>

  override def transform[C: ClassTag](transformer: Transformer[T, C]): DataSet[C] = {
    // val preDataSet = this

    new LocalDataSet[C] {

      override def data(train: Boolean): Iterator[C] = transformer(self.data(train))

      override def shuffle(): Unit = self.shuffle()

      override def size(): Long = self.size()
    }
  }
}

// class LocalArrayDataSet[T, C](buffer: Array[T], tt: Transformer[T, C]) extends AbstractDataSet[T, Iterator[C]] {
private[dataset] class LocalArrayDataSet[T](buffer: Array[T]) extends LocalDataSet[T] {

  override def shuffle(): Unit = {
    DataUtil.shuffle(buffer)
  }

  override def data(train: Boolean): Iterator[T] = {
    new Iterator[T] {
      private val index = new AtomicInteger()

      override def hasNext: Boolean = {
        if (train) {
          true
        } else {
          index.get() < buffer.length
        }
      }

      override def next(): T = {
        val curIndex = index.getAndIncrement()
        if (train || curIndex < buffer.length) {
          val i = if (train) curIndex % buffer.length else curIndex
          buffer(i)
        } else {
          null.asInstanceOf[T]
        }
      }
    }
  }

  override def size(): Long = buffer.length
}

trait DistributedDataSet[T] extends AbstractDataSet[T, RDD[T]] { self =>

  var isCached = false

  override def transform[C: ClassTag](transformer: Transformer[T, C]): DataSet[C] = {
    // val preDataSet = this
    val broadcast = this.originRDD().sparkContext.broadcast(transformer)
    val cachedTransformer: RDD[Transformer[T, C]] = self.originRDD().mapPartitions(_ =>
      Iterator.single(broadcast.value.cloneTransformer())
    ).setName("Cached Transformer").persist()

    new DistributedDataSet[C] {
      override def originRDD(): RDD[_] = self.originRDD()

      override def data(train: Boolean): RDD[C] = {
        self.data(train).zipPartitions(cachedTransformer)((data, tran) => tran.next()(data))
      }

      override def shuffle(): Unit = self.shuffle()

      override def size(): Long = self.size()

      override def cache(): Unit = {
        cachedTransformer.count()
        isCached = true
      }

      override def unpersist(): Unit = {
        cachedTransformer.unpersist()
        isCached = false
      }
    }
  }

  def originRDD(): RDD[_]

  def cache(): Unit = {
    if (originRDD() != null) {
      originRDD().count()
    }
    isCached = true
  }

  def unpersist(): Unit = {
    if (originRDD() != null) {
      originRDD().unpersist()
      isCached = false
    }
  }
}

private[dataset] class CachedDistriDataSet[T: ClassTag](buffer: RDD[Array[T]])
    extends DistributedDataSet[T] {

  protected lazy val count: Long = buffer.mapPartitions(iter => {
    require(iter.hasNext)
    val array = iter.next()
    require(!iter.hasNext)
    Iterator.single(array.length)
  }).reduce(_ + _)

  protected var indexes: RDD[Array[Int]] = buffer.mapPartitions(iter => {
    Iterator.single(iter.next().indices.toArray)
  }).setName("original index").cache()

  override def data(train: Boolean): RDD[T] = {
    val generator = new RandomDataGenerator()
    val _train = train
    buffer.zipPartitions(indexes)((dataIter, indexIter) => {
      val indexes: Array[Int] = indexIter.next()
      val indexOffset = math.max(1, indexes.length)
      val localData: Array[T] = dataIter.next()
      val offset = if (_train) generator.nextUniform(0, indexOffset).toInt else 0

      new Iterator[T] {
        private val _offset = new AtomicInteger(offset)

        override def hasNext: Boolean = {
          if (_train) true else _offset.get() < localData.length
        }

        override def next(): T = {
          val i = _offset.getAndIncrement()
          if (_train) {
            localData(indexes(i % localData.length))
          } else {
            if (i < localData.length) {
              localData(indexes(i))
            } else {
              null.asInstanceOf[T]
            }
          }
        }
      }
    })
  }

  override def size(): Long = count

  override def shuffle(): Unit = {
    indexes.unpersist()
    indexes = buffer.mapPartitions(iter => {
      Iterator.single(DataUtil.shuffle(iter.next().indices.toArray))
    }).setName("shuffled index").cache()
  }

  override def originRDD(): RDD[_] = buffer

  override def cache(): Unit = {
    buffer.count()
    indexes.count()
    isCached = true
  }

  override def unpersist(): Unit = {
    buffer.unpersist()
    indexes.unpersist()
    isCached = false
  }
}

object DataSet {

  def array[T](data: Array[T]): LocalArrayDataSet[T] = {
    new LocalArrayDataSet[T](data)
  }

  def array[T: ClassTag](localData: Array[T], sc: SparkContext): DistributedDataSet[T] = {
    val nodeNumber = Engine.nodeNumber()
    new CachedDistriDataSet[T](
      sc.parallelize(localData, nodeNumber)
        .coalesce(nodeNumber, shuffle = true)
        .mapPartitions(iter => Iterator.single(iter.toArray))
        .setName("cached dataset")
        .cache())
  }

  def rdd[T: ClassTag](data: RDD[T], partitionNum: Int = Engine.nodeNumber()): DistributedDataSet[T] = {
    new CachedDistriDataSet[T](
      data.coalesce(partitionNum, shuffle = true)
        .mapPartitions(iter => Iterator.single(iter.toArray))
        .setName("cached dataset")
        .cache()
    )
  }
}
