resolvers ++= Seq(
  "Sonatype Releases"   at "https://oss.sonatype.org/content/repositories/releases/",
  "Spark Packages repo" at "https://dl.bintray.com/spark-packages/maven/"
)

addSbtPlugin("org.spark-packages" %% "sbt-spark-package" % "0.2.6")

addSbtPlugin("org.scoverage"     % "sbt-scoverage"       % "1.5.0")

addSbtPlugin("com.lucidchart"    % "sbt-scalafmt"        % "1.10")
