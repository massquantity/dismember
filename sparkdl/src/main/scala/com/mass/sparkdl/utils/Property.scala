package com.mass.sparkdl.utils

import java.io.InputStream
import java.nio.file.{Files, Paths}

import scala.collection.mutable

import com.mass.sparkdl.{getScalaVersion, getSparkVersion}
import org.apache.log4j.Logger
import org.apache.spark.{SparkConf, SparkContext, SparkException}

object Property {

  val logger: Logger = Logger.getLogger(getClass)

  def readConf(
      path: String,
      prefix: String,
      truncate: Boolean = true): mutable.LinkedHashMap[String, String] = {

    val stream: InputStream = getClass.getResourceAsStream(path)
    val lines = scala.io.Source.fromInputStream(stream)
      .getLines
      .toSeq
      .filter(_.startsWith(prefix))
      .map(_.trim.split("\\s+"))
      .filter(_.length == 2)
      //  require(line.length == 2, s"""missing value in "${line.mkString}"""")
      .map(line => {
        val key = if (truncate) line(0).substring(prefix.length + 1) else line(0)
        val value = line(1)
        (key, value)
      })

    mutable.LinkedHashMap[String, String](lines: _*)
  }

  def createSparkConf(conf: Map[String, String], existingConf: Option[SparkConf] = None): SparkConf = {
    val sparkConf = existingConf match {
      case Some(e) => e
      case None => new SparkConf()
    }
    conf.foreach(i => sparkConf.set(i._1, i._2))
    sparkConf
  }

  def configLocal(conf: Map[String, String]): Unit = {
    logger.info(s"Scala version: $getScalaVersion. " +
      "Detected localMode. Run workload without spark")
    val nodeNum = 1
    val coreNum = getCoreNumber(conf("thread_number").toInt)
    logger.info(s"Total thread num: $coreNum")
    Engine.setNodeAndCore(nodeNum, coreNum)
  }

  def configDist(conf: Map[String, String]): SparkContext = {
    logger.info(s"Scala version: $getScalaVersion, Spark version: $getSparkVersion. " +
      s"Using spark mode")
    val sc = new SparkContext(createSparkConf(conf))
    val (nExecutor, executorCores) = sparkExecutorAndCore().get
    logger.info(s"Executor number: $nExecutor, executor cores number: $executorCores")
    Engine.setNodeAndCore(nExecutor, executorCores)
    sc
  }

  private def getCoreNumber(confNum: Int): Int = {
    if (confNum <= 0) {
      val coreNum = Runtime.getRuntime.availableProcessors()
      // if (coreNum > 1) coreNum / 2 else 1
      coreNum
    } else {
      confNum
    }
  }

  private def sparkExecutorAndCore(): Option[(Int, Int)] = {
    try {
      parseExecutorAndCore(SparkContext.getOrCreate().getConf)
    } catch {
      case s: SparkException =>
        if (s.getMessage.contains("A master URL must be set in your configuration")) {
          throw new IllegalArgumentException("A master URL must be set in your configuration." +
            " Or you can run in a local JVM environment by using LocalMode")
        }
        throw s
    }
  }

  //noinspection RegExpRedundantEscape
  private def parseExecutorAndCore(conf: SparkConf): Option[(Int, Int)] = {
    val master = conf.get("spark.master", null)

    if (master.toLowerCase.startsWith("local")) {
      // Spark local mode
      val patternLocalN = "local\\[(\\d+)\\]".r
      val patternLocalStar = "local\\[\\*\\]".r
      master match {
        case patternLocalN(n) => Some(1, n.toInt)
        case patternLocalStar(_*) => Some(1, Runtime.getRuntime.availableProcessors())
        case _ => throw new IllegalArgumentException(s"Can't parse master $master")
      }
    } else if (master.toLowerCase.startsWith("spark")) {
      // Spark standalone mode
      val coreString = conf.get("spark.executor.cores", null)
      val maxString = conf.get("spark.cores.max", null)
      require(coreString != null, "must set spark.executor.cores property")
      require(maxString != null, "must set spark.cores.max property")
      val coreNum = coreString.toInt
      val nodeNum = {
        val total = maxString.toInt
        require(total >= coreNum && total % coreNum == 0,
          "total core must be divided by single core number")
        total / coreNum
      }
      Some(nodeNum, coreNum)
    } else if (master.toLowerCase.startsWith("yarn")) {
      val coreString = conf.get("spark.executor.cores", null)
      require(coreString != null, "must set spark.executor.cores property")
      val coreNum = coreString.toInt
      val nodeNum = {
        val numExecutorString = conf.get("spark.executor.instances", null)
        require(numExecutorString != null, "must set spark.executor.instances property")
        numExecutorString.toInt
      }
      Some(nodeNum, coreNum)
    } else {
      throw new IllegalArgumentException(s"Unsupported master format $master")
    }
  }

  def getOrStop(conf: Map[String, String], key: String): String = {
    conf.getOrElse(key, throw new IllegalArgumentException(
      s"failed to read parameter: $key in conf file"))
  }

  def exists(path: String): Boolean = {
    Files.exists(Paths.get(path))
  }

  def getOption[T](conf: Map[String, String], key: String): Option[T] = {
    conf.get(key) match {
      case Some(i) => Some(i.asInstanceOf[T])
      case None => None
    }
  }
}
