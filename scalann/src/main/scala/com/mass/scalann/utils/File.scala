package com.mass.scalann.utils

import java.io._
import java.net.URI

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, FSDataInputStream, FSDataOutputStream, Path}
import org.apache.hadoop.io.IOUtils

object File {

  private[scalann] val hdfsPrefix: String = "hdfs:"

  def save(obj: Serializable, fileName: String, isOverwrite: Boolean): Unit = {
    var fw: FileWriter = null
    var out: OutputStream = null
    var objFile: ObjectOutputStream = null
    try {
      fw = FileWriter(fileName)
      out = fw.create(isOverwrite)
      objFile = new ObjectOutputStream(new BufferedOutputStream(out))
      objFile.writeObject(obj)
    } finally {
      if (null != objFile) objFile.close()
      if (null != out) out.close()
      if (null != fw) fw.close()
    }
  }

  def saveBytes(bytes: Array[Byte], fileName: String, isOverwrite: Boolean = false): Unit = {
    var fw: FileWriter = null
    var out: OutputStream = null
    try {
      fw = FileWriter(fileName)
      out = fw.create(isOverwrite)
      IOUtils.copyBytes(new ByteArrayInputStream(bytes), out, 1024, true)
    } finally {
      if (null != out) out.close()
      if (null != fw) fw.close()
    }
  }

  private[scalann] def getFileSystem(fileName: String): FileSystem = {
    val src = new Path(fileName)
    val fs = src.getFileSystem(File.getConfiguration(fileName))
    require(fs.exists(src), src.toString + " does not exists")
    fs
  }

  private[scalann] def getConfiguration(fileName: String): Configuration = {
    if (fileName.startsWith(File.hdfsPrefix)) {
      new Configuration()
    } else {
      new Configuration(false)
    }
  }

  def saveToHdfs(obj: Serializable, fileName: String, overwrite: Boolean): Unit = {
    require(fileName.startsWith(File.hdfsPrefix), s"hdfs path $fileName should have prefix 'hdfs:'")
    val dest = new Path(fileName)
    var fs: FileSystem = null
    var out: FSDataOutputStream = null
    var objFile: ObjectOutputStream = null
    try {
      fs = dest.getFileSystem(new Configuration())
      if (fs.exists(dest)) {
        if (overwrite) {
          fs.delete(dest, true)
        } else {
          throw new RuntimeException(s"file $fileName already exists")
        }
      }
      out = fs.create(dest)
      val byteArrayOut = new ByteArrayOutputStream()
      objFile = new ObjectOutputStream(byteArrayOut)
      objFile.writeObject(obj)
      IOUtils.copyBytes(new ByteArrayInputStream(byteArrayOut.toByteArray), out, 1024, true)
    } finally {
      if (null != objFile) objFile.close()
      if (null != out) out.close()
      if (null != fs) fs.close()
    }
  }

  def loadFromHdfs[T](fileName: String): T = {
    val byteArrayOut = readHdfsByte(fileName)
    var objFile: ObjectInputStream = null
    try {
      objFile = new ObjectInputStream(new ByteArrayInputStream(byteArrayOut))
      val result = objFile.readObject()
      objFile.close()
      result.asInstanceOf[T]
    } finally {
      if (null != objFile) objFile.close()
    }
  }

  def load[T](fileName: String): T = {
    var fr: FileReader = null
    var in: InputStream = null
    var objFile: ObjectInputStream = null
    try {
      fr = FileReader(fileName)
      in = fr.open()
      val bis = new BufferedInputStream(in)
      objFile = new ObjectInputStream(bis)
      objFile.readObject().asInstanceOf[T]
    } finally {
      if (null != in) in.close()
      if (null != fr) fr.close()
      if (null != objFile) objFile.close()
    }
  }

  def readBytes(fileName: String): Array[Byte] = {
    var fr: FileReader = null
    var in: InputStream = null
    try {
      fr = FileReader(fileName)
      in = fr.open()
      val byteArrayOut = new ByteArrayOutputStream()
      IOUtils.copyBytes(in, byteArrayOut, 1024, true)
      byteArrayOut.toByteArray
    } finally {
      if (null != in) in.close()
      if (null != fr) fr.close()
    }
  }

  def readHdfsByte(fileName: String): Array[Byte] = {
    val src: Path = new Path(fileName)
    var fs: FileSystem = null
    var in: FSDataInputStream = null
    try {
      fs = FileSystem.newInstance(new URI(fileName), new Configuration())
      in = fs.open(src)
      val byteArrayOut = new ByteArrayOutputStream()
      IOUtils.copyBytes(in, byteArrayOut, 1024, true)
      byteArrayOut.toByteArray
    } finally {
      if (null != in) in.close()
      if (null != fs) fs.close()
    }
  }
}

class FileReader(fileName: String) {
  private var inputStream: InputStream = _
  private val conf = File.getConfiguration(fileName)
  private val path = new Path(fileName)
  private val fs: FileSystem = path.getFileSystem(conf)

  def open(): InputStream = {
    require(fs.exists(path), s"$fileName is empty!")
    inputStream = fs.open(path)
    inputStream
  }

  def close(): Unit = {
    if (null != inputStream) inputStream.close()
    fs.close()
  }
}

object FileReader {
  def apply(fileName: String): FileReader = {
    new FileReader(fileName)
  }
}

class FileWriter(fileName: String) {
  private var outputStream: OutputStream = _
  private val conf = File.getConfiguration(fileName)
  private val path = new Path(fileName)
  private val fs: FileSystem = path.getFileSystem(conf)
  fs.setWriteChecksum(false)

  def create(overwrite: Boolean = false): OutputStream = {
    if (!overwrite) {
      require(!fs.exists(path), s"$fileName already exists!")
    }
    outputStream = fs.create(path, overwrite)
    outputStream
  }

  def close(): Unit = {
    if (null != outputStream) outputStream.close()
    fs.close()
  }
}

object FileWriter {
  def apply(fileName: String): FileWriter = {
    new FileWriter(fileName)
  }
}
