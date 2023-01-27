package com.mass.scalann.utils

import java.util.concurrent._

import scala.collection.mutable
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.concurrent.duration.Duration
import scala.jdk.CollectionConverters._

import com.intel.analytics.bigdl.mkl.MKL
import org.apache.commons.lang3.exception.ExceptionUtils
import org.apache.log4j.Logger

class ThreadPool(_poolSize: Int) {
  import ThreadPool._

  private var poolSize = _poolSize
  private var mklPoolSize: Option[Int] = None
  private var threadPool: ExecutorService = _

  private var context: ExecutionContext = spawnThreadPool(poolSize)

  private type JavaFuture[T] = java.util.concurrent.Future[T]

  private def spawnThreadPool(poolSize: Int): ExecutionContext = {
    if (poolSize == 1) {
      threadPool = Executors.newFixedThreadPool(
        poolSize,
        new ThreadFactory {
          override def newThread(r: Runnable): Thread = {
            val t = Executors.defaultThreadFactory().newThread(r)
            t.setName("single-thread-computing")
            t.setDaemon(true)
            t
          }
        }
      )
      singleThreadPool
    } else {
      new ExecutionContext {
        if (threadPool != null) {
          threadPool.shutdown()
        }
        threadPool = Executors.newFixedThreadPool(
          poolSize,
          (r: Runnable) => {
            val t = Executors.defaultThreadFactory().newThread(r)
            t.setName("default-thread-computing " + t.getId)
            t.setDaemon(true)
            t
          }
        )

        override def execute(runnable: Runnable): Unit = {
          threadPool.submit(runnable)
        }

        override def reportFailure(cause: Throwable): Unit = {}
      }
    }
  }

  def getPoolSize: Int = poolSize

  def setPoolSize(size: Int): this.type = synchronized {
    if (size != poolSize) {
      context = spawnThreadPool(size)
      poolSize = size
      if (mklPoolSize.isDefined) {
        setMKLThread(mklPoolSize.get)
      }
    }
    this
  }

  def setMKLThread(size: Int): this.type = synchronized {
    require(MKL.isMKLLoaded)
    mklPoolSize = Some(size)
    (1 to poolSize) map { _ =>
      Future {
        MKL.setNumThreads(size)
        val tid = Thread.currentThread().getId
        // logger.info(s"Set mkl threads to $size on thread $tid")
      }(context)
    } foreach (Await.result(_, Duration.Inf))
    this
  }

  def invoke2[T](tasks: Seq[() => T]): Seq[JavaFuture[T]] = {
    tasks
      .map { task =>
        new Callable[T] {
          override def call(): T = {
            try {
              task()
            } catch {
              case t: Throwable =>
                logger.error("Error: " + ExceptionUtils.getStackTrace(t))
                throw t
            }
          }
        }
      }
      .map(threadPool.submit(_))
  }

  def invoke[T](tasks: Seq[() => T]): Seq[Future[T]] = {
    tasks.map(task =>
      Future {
        try {
          task()
        } catch {
          case t: Throwable =>
            logger.error("Error: " + ExceptionUtils.getStackTrace(t))
            throw t
        }
      }(context)
    )
  }

  def invoke[T](task: () => T): Future[T] = {
    Future {
      try {
        task()
      } catch {
        case t: Throwable =>
          logger.error("Error: " + ExceptionUtils.getStackTrace(t))
          throw t
      }
    }(context)
  }

  def invokeAndWait[T](tasks: Seq[() => T], timeout: Duration = Duration.Inf): Seq[T] = {
    tasks
      .map(task =>
        Future {
          try {
            task()
          } catch {
            case t: Throwable =>
              logger.error("Error: " + ExceptionUtils.getStackTrace(t))
              throw t
          }
        }(context)
      )
      .map(future => {
        Await.result(future, timeout)
      })
  }

  def invokeAndWait2[T](
      tasks: Seq[() => T],
      timeout: Long = Long.MaxValue,
      timeUnit: TimeUnit = TimeUnit.NANOSECONDS
  ): mutable.Buffer[JavaFuture[T]] = {
    val callables = tasks.map(task =>
      new Callable[T] {
        override def call(): T = {
          task()
        }
      }
    )

    val resultFutures = threadPool.invokeAll(callables.asJava, timeout, timeUnit)
    var i = 0
    while (i < resultFutures.size()) {
      try {
        resultFutures.get(i).get()
      } catch {
        case t: ExecutionException => throw t.getCause
        case i: InterruptedException => throw i.getCause
      }
      i += 1
    }
    resultFutures.asScala
  }

  def sync(futures: Seq[Future[_]], timeout: Duration = Duration.Inf): Unit = {
    futures.foreach { f =>
      {
        Await.result(f, timeout)
      }
    }
  }
}

object ThreadPool {
  val singleThreadPool: ExecutionContext = new ExecutionContext {
    override def execute(runnable: Runnable): Unit = {
      runnable.run()
    }

    override def reportFailure(cause: Throwable): Unit = {}
  }

  private val logger = Logger.getLogger(getClass)
}
