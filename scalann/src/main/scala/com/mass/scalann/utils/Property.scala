package com.mass.scalann.utils

import java.nio.file.{Files, Paths}

import com.mass.scalann.getScalaVersion
import org.apache.log4j.Logger

object Property {

  val logger: Logger = Logger.getLogger(getClass)

  def readConf(
    path: String,
    prefix: String,
    name: String,
    truncate: Boolean = true,
    print: Boolean
  ): Map[String, String] = {
    val fileSource = path match {
      case "fromResource" =>
        logger.info("Using config file from resources...")
        scala.io.Source.fromInputStream(getClass.getResourceAsStream(s"/$name.conf"))
      case _ =>
        logger.info("Using user defined config file...")
        require(fileExists(path), s"Config file $path doesn't exist")
        scala.io.Source.fromFile(path)
    }

    val lines = fileSource
      .getLines()
      .toSeq
      .filter(_.startsWith(prefix))
      .map(_.trim.split("\\s+"))
      .filter(_.length == 2)
      .map(line => {
        val key = if (truncate) line(0).substring(prefix.length + 1) else line(0)
        val value = line(1)
        (key, value)
      })

    if (print) {
      logger.info(s"${"=" * 36} $prefix configs ${"=" * 38}")
      lines.foreach(i => logger.info(s"${i._1} = ${i._2}"))
      println()
    }

    // mutable.LinkedHashMap[String, String](lines: _*)
    Map(lines: _*)
  }

  def configLocal(conf: Map[String, String]): Unit = {
    val coreNum = getCoreNumber(conf("thread_number").toInt)
    logger.info(s"Scala version: $getScalaVersion, total thread num: $coreNum")
    Engine.setCoreNumber(coreNum)
  }

  private def getCoreNumber(confNum: Int): Int = {
    if (confNum <= 0) {
      val coreNum = Runtime.getRuntime.availableProcessors()
      coreNum
    } else {
      confNum
    }
  }

  def getOrStop(conf: Map[String, String], key: String): String = {
    conf.getOrElse(key, throw new IllegalArgumentException(
      s"failed to read parameter: $key in conf file"))
  }

  def fileExists(path: String): Boolean = {
    Files.exists(Paths.get(path))
  }

  def filePath(moduleName: String): String = {
    val prefix = System.getProperty("user.dir")
    if (prefix.contains(moduleName)) {
      Paths.get(prefix).getParent.toString
    } else {
      prefix
    }
  }

  def getOption[T](conf: Map[String, String], key: String): Option[T] = {
    conf.get(key) match {
      case Some(i) => Some(i.asInstanceOf[T])
      case None => None
    }
  }
}
