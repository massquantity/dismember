import sbt._

object Dependencies {

  val scalaTest = Seq(
    "org.scalatest" %% "scalatest" % "3.2.12" % Test
  )

  val scalapb = Seq(
    "com.thesamet.scalapb" %% "scalapb-runtime" % "0.10.11"
  )

  val scopt = Seq(
    "com.github.scopt" %% "scopt" % "4.0.1"
  )

  val commonMath = Seq(
    "org.apache.commons" % "commons-math3" % "3.6.1"
  )

  val smile = Seq(
    "com.github.haifengl" % "smile-core" % "2.6.0"
  )

  val blasLinux = Seq(
    "org.bytedeco" % "javacpp"   % "1.5.4"        classifier "linux-x86_64",
    "org.bytedeco" % "openblas"  % "0.3.10-1.5.4" classifier "linux-x86_64",
    "org.bytedeco" % "arpack-ng" % "3.7.0-1.5.4"  classifier "linux-x86_64",
  )

  val mklLinux = Seq(
    "com.intel.analytics.bigdl.core.native.mkl" % "mkl-java-x86_64-linux" % "2.0.0"
  )

  val hadoop = Seq(
    "org.apache.hadoop" % "hadoop-common" % "3.3.1"
  )

  val commonDependencies = scalaTest ++ scalapb ++ scopt ++ commonMath

  val dlDependencies = mklLinux ++ hadoop

  val tdmDependencies = smile ++ blasLinux

}