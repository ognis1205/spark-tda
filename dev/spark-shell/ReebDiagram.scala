/*
 * Computes Reeb Diagram
 * 
 * SAMPLE DATA:
 *   15.55	28.65
 *   14.9	27.55
 *   14.45	28.35
 *   14.15	28.8
 *   13.75	28.05
 *
 * USAGE:
 *   - To load this file from spark-shell simply do
 *     spark> :load PATH_TO_THIS_FILE
 */
import java.io.{File, PrintWriter}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.{ReebDiagram, VectorAssembler}
import org.apache.spark.sql.functions._

def computeReebDiagram(
  pathToTextFile: String,
  quantity: Int,
  linkThresholdRatio: Double,
  coreThresholdRatio: Double,
  topTreeRatio: Double) {

  def save(f: File)(func: PrintWriter => Unit) {
    val p = new PrintWriter(f)
    try {
      func(p)
    } finally {
      p.close()
    }
  }

  val filename = pathToTextFile.split("\\.")(0)

  val outputFilename = s"$filename-REEB-k${quantity}-l${linkThresholdRatio}-c${coreThresholdRatio}-i${topTreeRatio}.tsv"

  val points = sc.textFile(pathToTextFile)
    .map {
      line => line.trim.split("\\s+")
    }
    .zipWithIndex
    .map { case (row, i) =>
      (i, row(0).toDouble, row(1).toDouble, 0)
    }
    .toDF("id", "x", "y", "cover_id")

  val cardinality = points.count

  val assembler = new VectorAssembler()
    .setInputCols(Array("x", "y"))
    .setOutputCol("feature")

  val features = assembler
    .transform(points)

  val reeb = new ReebDiagram()
    .setK(quantity)
    .setLinkThresholdRatio(linkThresholdRatio)
    .setCoreThresholdRatio(coreThresholdRatio)
    .setTopTreeSize((topTreeRatio * cardinality).toInt)
    .setTopTreeLeafSize(quantity)
    .setIdCol("id")
    .setCoverCol("cover_id")
    .setFeaturesCol("feature")
    .setOutputCol("cluster_id")

  val transformed = reeb
    .fit(features)
    .transform(features)

  val clusters = Map(
    transformed
      .select("cluster_id")
      .rdd
      .map(row => row.getLong(0))
      .distinct
      .zipWithIndex
      .collect(): _*)

  val result = transformed
    .select("x", "y", "cluster_id")
    .rdd
    .map(row => (row.getDouble(0), row.getDouble(1), row.getLong(2)))
    .map { case (x, y, clusterId) => (x, y, clusters(clusterId) + 1)}
    .collect()

  save(new File(outputFilename)) {
    println(s"OUTPUT TO: ${outputFilename}")
    f => result.foreach{
      case (x, y, ccid) => f.println(s"${x}\t${y}\t${ccid}")
    }
  }
}
