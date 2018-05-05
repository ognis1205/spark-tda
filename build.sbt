// Versions of depending packages.
lazy val versions = new {
  val `scalacheck` = "1.13.2"
  val `scalatest` = "3.0.3"
  val `spark` = "2.2.0"
  val `spark-testing-base` = "0.8.0"
  val `specs` = "3.8.9"
  val `specs2` = "2.4.17"
}

// Target Spark version.
val targetSparkVersion = sys.props.getOrElse("spark.version", versions.`spark`)

val sparkBranch = targetSparkVersion.split("\\.").take(2).mkString(".")

val defaultScalaVersion = sparkBranch match {
  case "2.0" => "2.11.11"
  case "2.1" => "2.11.11"
  case "2.2" => "2.11.11"
  case _ =>
    throw new IllegalArgumentException(
      s"Unsupported Spark version: $targetSparkVersion.")
}

// Target Scala version
val targetScalaVersion =
  sys.props.getOrElse("scala.version", defaultScalaVersion)

val scalaBranch = targetScalaVersion.split("\\.").take(2).mkString(".")

// Spark dependencies.
lazy val sparkDependencies = Seq("graphx", "mllib", "sql")

// Unit test dependencies.
lazy val unitTestDependencies = Seq(
  "org.scalatest" %% "scalatest" % versions.`scalatest` % "test",
  "org.specs2" %% "specs2-core" % versions.`specs` % "test",
  "org.specs2" %% "specs2-mock" % versions.`specs2` % "test",
  "com.holdenkarau" %% "spark-testing-base" % Seq(
    targetSparkVersion,
    versions.`spark-testing-base`).mkString("_") % "test",
  "org.scalacheck" %% "scalacheck" % versions.`scalacheck` % "test"
)

sparkVersion := targetSparkVersion

scalaVersion := targetScalaVersion

name := "spark-tda"

organization := "ognis1205"

spName := s"ognis1205/spark-tda"

// Don't forget to set the version.
// See: https://github.com/databricks/sbt-spark-package/issues/17
version := s"0.0.1-SNAPSHOT-spark$sparkBranch-s_$scalaBranch"

// All Spark Packages need a license.
licenses := Seq(
  "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0"))

spShortDescription := "Topological Data Analysis Package"

spDescription := """SparkTDA is a package for Apache Spark providing Topological Data Analysis Functionalities.
                    |
                    | The current plan is to implement the following features:
                    |
                    | - Scalable Mapper as ML Pipeline
                    | - Scalable Multiscale Mapper as ML Pipeline
                    | - Persistent Homology Computation of Towers of Simplicial Complexes
                    |
                 """.stripMargin

spHomepage := "https://github.com/ognis1205/spark-tda"

credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials")

// See: https://github.com/databricks/sbt-spark-package/issues/17
crossPaths := false

// See: https://github.com/databricks/sbt-spark-package/issues/17
spAppendScalaVersion := false

// Add Spark components this package depends on, e.g, "mllib", ....
sparkComponents ++= sparkDependencies

// Uncomment and change the value below to change the directory where your zip artifact will be created
// spDistDirectory := target.value

// Add any Spark Package dependencies using spDependencies.
// e.g. spDependencies += "databricks/spark-avro:0.1"

libraryDependencies ++= unitTestDependencies

parallelExecution := false

scalacOptions := Seq("-feature",
                     "-deprecation",
                     "-Xfatal-warnings",
                     "-Ypatmat-exhaust-depth",
                     "off")

// This fixes a class loader problem with scala.Tuple2 class, scala-2.11, Spark 2.x
fork in Test := true

// This and the next line fix a problem with forked run: https://github.com/scalatest/scalatest/issues/770
//javaOptions in Test ++= Seq("-Xmx2048m",
//                            "-XX:ReservedCodeCacheSize=384m",
//                            "-XX:MaxPermSize=384m")

concurrentRestrictions in Global := Seq(Tags.limitAll(1))

autoAPIMappings := true

coverageHighlighting := false

scalafmtOnCompile := true
