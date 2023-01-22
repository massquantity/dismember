import Dependencies._

ThisBuild / version := "0.2.0"
ThisBuild / scalaVersion := "2.13.8"
ThisBuild / githubWorkflowSbtCommand := "sbt -mem 4096"
ThisBuild / githubWorkflowArtifactUpload := false
ThisBuild / githubWorkflowPublishTargetBranches := Seq()
ThisBuild / githubWorkflowUseSbtThinClient := false

lazy val root = (project in file("."))
  .settings(name := "dismember")
  .settings(publish / skip := true)
  .aggregate(scalann, tdm, jtm, otm, `deep-retrieval`, examples)

lazy val scalann = (project in file("scalann"))
  .settings(name := "scalann")
  .settings(commonSettings)
  .settings(libraryDependencies ++= dlDependencies)

lazy val tdm = (project in file("tdm"))
  .settings(name := "tdm")
  .settings(commonSettings)
  .settings(libraryDependencies ++= tdmDependencies)
  .dependsOn(scalann)

lazy val jtm = (project in file("jtm"))
  .settings(name := "jtm")
  .settings(commonSettings)
  .settings(libraryDependencies ++= commonDependencies)
  .dependsOn(scalann, tdm)

lazy val otm = (project in file("otm"))
  .settings(name := "otm")
  .settings(commonSettings)
  .settings(libraryDependencies ++= commonDependencies)
  .dependsOn(scalann, tdm, jtm)

lazy val `deep-retrieval` = (project in file("deep-retrieval"))
  .settings(name := "deep-retrieval")
  .settings(commonSettings)
  .settings(libraryDependencies ++= commonDependencies)
  .dependsOn(scalann)

lazy val examples = (project in file("examples"))
  .enablePlugins(JavaAppPackaging, JavaServerAppPackaging)
  .settings(name := "examples")
  .settings(commonSettings)
  .settings(libraryDependencies ++= exampleDependencies)
  .dependsOn(scalann, tdm, jtm, otm, `deep-retrieval`)

lazy val commonSettings = Seq(
  javacOptions ++= Seq(
    "-encoding", "utf8",
    "-source", "11",
    "-target", "11"
  ),
  scalacOptions ++= Seq(
    "-encoding", "utf8",
    "-feature",
    "-deprecation",
    "-explaintypes",
    "-unchecked",
    "-language:higherKinds",
    "-Xfatal-warnings",
    "-Vimplicits",
    "-Vtype-diffs"
  ),
  resolvers ++= Resolver.sonatypeOssRepos("releases") :+ Resolver.typesafeRepo("releases"),
  makeBatScripts                 := Nil,
  Compile / mainClass            := None,
  fork                           := true,
  Test / parallelExecution       := false,
  ThisBuild / parallelExecution  := false,
  Test / testForkedParallel      := false,
  ThisBuild / testForkedParallel := false,
  Global / concurrentRestrictions += Tags.limit(Tags.Test, 1)
)
