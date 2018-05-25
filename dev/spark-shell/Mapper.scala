/*
 * Computes Mapper
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
import scala.collection.mutable.ArrayBuilder
import org.apache.spark.graphx.{Graph, EdgeRDD, VertexRDD, Edge}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.{Vectors, Vector}
import org.apache.spark.ml.feature.{Cover, ReebDiagram, PCA, MinMaxScaler}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

def computeMapper(
  pathToTextFile: String,
  numComponents: Int,
  numSplits: Int,
  overlapRatio: Double,
  quantity: Int,
  linkThresholdRatio: Double,
  coreThresholdRatio: Double,
  topTreeRatio: Double) = {

  def parse(line: Seq[String]): Vector = {
    val (indices, values) = (ArrayBuilder.make[Int], ArrayBuilder.make[Double])
    for (entry <- line) {
      val indexValue = entry.trim.split(":")
      indices += indexValue(0).toInt
      values += indexValue(1).toDouble
    }
    Vectors.sparse(784, indices.result(), values.result()).compressed
  }

  def save(f: File)(func: PrintWriter => Unit) {
    val p = new PrintWriter(f)
    try {
      func(p)
    } finally {
      p.close()
    }
  }

  def rgb(value: Double, minimum: Double = 0.0, maximum: Double = 1.0): (Int, Int, Int) = {
    val ratio = 2.0 * (value - minimum) / (maximum - minimum)
    val b = math.max(0.0, 255.0 * (1.0 - ratio)).toInt
    val r = math.max(0.0, 255.0 * (ratio - 1.0)).toInt
    val g = 255 - b - r
    (r, g, b)
  }

  def radius(value: Long): Double = {
    val scaled =
      if (value >= 50) 50.0
      else if (value < 20) 20.0
      else value.toDouble
    scaled / 50.0
  }

  def toGexf(graph: Graph[(Double, Long, String), Long]): String = {
    def toNodeTags(vertices: VertexRDD[(Double, Long, String)]): String =
      vertices
        .map { v =>
          val (id, attr) = (v._1, v._2)
          val (r, g, b) = rgb(attr._1)
          val size = radius(attr._2)
          val label = attr._3
          s"""<node id="$id" label="$label"><viz:color r="$r" g="$g" b="$b"/><viz:size value="$size"/></node>"""
        }
        .collect
        .mkString("\n")
    def toEdgeTags(edges: EdgeRDD[Long]): String =
      edges
        .map { e =>
          s"""<edge source="${e.srcId}" target="${e.dstId}" weight="${radius(e.attr) * 0.1}"><viz:thickness value="${radius(e.attr)}"/></edge>"""
        }
        .collect
        .mkString("\n")
    s"""<?xml version="1.0" encoding="UTF-8"?>
       |<gexf xmlns="http://www.gexf.net/1.1draft"
       |      version="1.1"
       |      xmlns:viz="http://www.gexf.net/1.1draft/viz"
       |      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       |      xsi:schemaLocation="http://www.gexf.net/1.1draft http://www.gexf.net/1.1draft/gexf.xsd">
       |  <graph defaultedgetype="undirected" timeformat="double" mode="dynamic">
       |    <nodes>
       |      ${toNodeTags(graph.vertices)}
       |    </nodes>
       |    <edges>
       |      ${toEdgeTags(graph.edges)}
       |    </edges>
       |  </graph>
       |</gexf>""".stripMargin
  }

  val filename = pathToTextFile.split("\\.")(0)

  val outputFilename = s"$filename-k${quantity}-s${numSplits}-l${linkThresholdRatio}-c${coreThresholdRatio}-i${topTreeRatio}.gexf"

  val points = sc.textFile(pathToTextFile)
    .map {
      line => line.trim.split("\\s+")
    }
    .map { row =>
      (row.head.toInt, parse(row.tail))
    }
    .toDF("label", "features")

  val cardinality = points.count

  val pca = new PCA()
    .setK(numComponents)
    .setInputCol("features")
    .setOutputCol("pca")

  val cover = new Cover()
    .setExploding(true)
    .setInputCols(pca.getOutputCol)
    .setNumSplits(numSplits)
    .setOverlapRatio(overlapRatio)
    .setOutputCol("cover_id")

  val reeb = new ReebDiagram()
    .setMeasure("euclidean")
    .setK(quantity)
    .setLinkThresholdRatio(linkThresholdRatio)
    .setCoreThresholdRatio(coreThresholdRatio)
    .setTopTreeSize((topTreeRatio * cardinality).toInt)
    .setTopTreeLeafSize(1000)
    .setSubTreeLeafSize(100)
    .setIdCol(cover.getIdCol)
    .setCoverCol(cover.getOutputCol)
    .setFeaturesCol("features")
    .setOutputCol("cluster_id")

  val mapper = new Pipeline()
    .setStages(Array(pca, cover, reeb))

  val firstElementUDF = udf { v: Vector => v(0) }

  println("EXECUTE MAPPER PIPELINE...")
  val reebDiagram = mapper
    .fit(points)
    .transform(points)

  println("COLLECT CLUSTER IDS...")
  val clusters = reebDiagram
    .groupBy("cluster_id")
    .count
    .select("cluster_id")
    .collect
    .filter(row => row.getLong(0) >= 0)
    .map(row => row.getLong(0))

  println("REMOVE NOISES...")
  val filtered = reebDiagram
    .filter(col("cluster_id").isin(clusters: _*))

  println("COLLECT DOMINANT LABELS IN EACH CLUSTERS...")
  val labels = scala.collection.mutable.Map[Long, String]()
  val histgram = filtered
    .groupBy("cluster_id", "label")
    .count
    .cache
  for ((cluster, i) <- clusters.zipWithIndex) {
    println(s"${i + 1} / ${clusters.length} CLUSTER ID: $cluster")
    val partialHistogram = histgram
      .where(s"cluster_id = $cluster")
      .select("label", "count")
    partialHistogram.show()
    val numLabels = partialHistogram.count
    val clusterSize = partialHistogram
      .agg(sum("count"))
      .first
      .getLong(0)
    val threshold = clusterSize.toDouble / numLabels.toDouble
    val label = partialHistogram
      .filter(s"count >= ${threshold.toInt}")
      .select("label")
      .collect
      .map(row => row.getInt(0))
      .sorted
      .mkString(",")
    labels += cluster -> label
  }

  println("GENERATE VERTICES...")
  val labelUDF = udf { v: Long => labels(v) }
  val scaler = new MinMaxScaler().setInputCol("pca").setOutputCol("scale")
  val scaled = scaler
    .fit(filtered)
    .transform(filtered)
    .drop("pca")

  val aggregated = scaled
    .withColumn("scale_tmp", firstElementUDF(scaled("scale")))
    .drop("scale")
    .withColumnRenamed("scale_tmp", "scale")
    .groupBy("cluster_id")
    .agg(mean("scale").alias("pca"), count(lit(1)).alias("cardinality"))

  val vertices = aggregated
    .withColumn("node_label", labelUDF(aggregated("cluster_id")))
    .select("cluster_id", "pca", "cardinality", "node_label")
    .rdd
    .map(row => (row.getLong(0), (row.getDouble(1), row.getLong(2), row.getString(3))))

  println("GENERATE EDGES...")
  val edges = filtered
    .groupBy("id")
    .agg(collect_list("cluster_id").alias("cluster_ids"))
    .select("cluster_ids")
    .rdd
    .flatMap(row => row.getAs[Seq[Long]](0).sorted.combinations(2))
    .toDF("edge")
    .groupBy("edge")
    .agg(count(lit(1)).alias("cardinality"))
    .select("edge", "cardinality")
    .rdd
    .map { row =>
      val srcDst = row.getAs[Seq[Long]](0)
      new Edge(srcDst(0), srcDst(1), row.getLong(1))
    }

  println(s"OUTPUT TO: ${outputFilename}")
  val graph = Graph(vertices, edges)
  save(new File(outputFilename)) { f =>
    f.println(toGexf(graph))
  }
}
