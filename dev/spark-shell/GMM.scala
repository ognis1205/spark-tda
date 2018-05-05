/*
 * Computes Gaussian Mixture Model
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
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.clustering.GaussianMixture
import org.apache.spark.sql.functions._

def computeGaussianMixtureModel(
  pathToTextFile: String,
  quantity: Int) {
  case class Point(x: Double, y: Double)

  def save(f: File)(func: PrintWriter => Unit) {
    val p = new PrintWriter(f)
    try {
      func(p)
    } finally {
      p.close()
    }
  }

  val filename = pathToTextFile.split("\\.")(0)

  val outputFilename = s"$filename-GMM-k${quantity}.tsv"

  val points = sc
    .textFile(pathToTextFile)
    .map {
      line => line.trim.split("\\s+")
    }
    .map {
      row => Point(row(0).toDouble, row(1).toDouble)
    }

  val features = points
    .map {
      p => Vectors.dense(p.x, p.y)
    }

  features.cache()

  val gmm = new GaussianMixture()
    .setK(quantity)
    .run(features)

  val predictions = features
    .map {
      f => (f(0), f(1), gmm.predict(f) + 1)
    }
    .collect

  save(new File(outputFilename)) {
    println(s"OUTPUT TO: ${outputFilename}")
    f => predictions.foreach{
      case (x, y, ccid) => f.println(s"${x}\t${y}\t${ccid}")
    }
  }
}
