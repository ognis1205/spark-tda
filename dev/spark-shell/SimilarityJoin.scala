/*
 * Computes SNN-DBSCAN
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
import org.apache.spark.graphx.lib.SharedNearestNeighbor
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}

def computeSimilarityJoin(
  pathToTextFile: String,
  joinType: SharedNearestNeighbor.JoinType,
  measure: SharedNearestNeighbor.Measure,
  quantity: Int,
  thresholdRatio:Double,
  signatureLength: Int = 2,
  numTables: Int = 1,
  bucketWidth: Double = 10.0) {
  case class Point(x: Double, y: Double)

  def save(f: File)(func: PrintWriter => Unit) {
    val p = new PrintWriter(f)
    try {
      func(p)
    } finally {
      p.close()
    }
  }

  val conf = SharedNearestNeighbor.Conf(
    joinType,
    measure,
    quantity,
    thresholdRatio
  )
  conf.setDimension(2)
  conf.setLenSignature(signatureLength)
  conf.setNumTables(numTables)
  conf.setBucketWidth(bucketWidth)

  val filename = pathToTextFile.split("\\.")(0)

  val outputFilename = if (conf.joinType == SharedNearestNeighbor.KNN_JOIN) {
    s"$filename-${conf.joinType}-${conf.measure}-k${conf.quantity}-r${conf.thresholdRatio}.tsv"
  } else {
    s"$filename-${conf.joinType}-${conf.measure}-k${conf.quantity}-r${conf.thresholdRatio}-s${conf.lenSignature.get}-t${conf.numTables.get}-b${conf.bucketWidth.get}.tsv"
  }

  val points = sc
    .textFile(pathToTextFile)
    .map {
      line => line.trim.split("\\s+")
    }
    .map {
      row => Point(row(0).toDouble, row(1).toDouble)
    }

  val indexedPoints = points
    .zipWithIndex
    .map {
      case (p, i) => (i, p)
    }

  val features = new IndexedRowMatrix(
    indexedPoints
      .map {
        case (i, p) => new IndexedRow(i, Vectors.dense(p.x, p.y))
      }
  )

  features.rows.cache()

  val connectedComponents = SharedNearestNeighbor
    .run(features, conf)
    .connectedComponents

  val connectedPoints = indexedPoints
    .join(connectedComponents.vertices)
    .map {
      case (id, (p, ccid)) => (id, p.x, p.y, ccid)
    }

  val clusters = Map(
    connectedPoints
      .map {
        case (_, _, _, ccid) => ccid
      }
      .distinct
      .zipWithIndex
      .collect
        : _*
  )

  val result = connectedPoints
    .map {
      case (_, x, y, ccid) => (x, y, clusters(ccid) + 1)
    }
    .collect

  save(new File(outputFilename)) {
    println(s"OUTPUT TO: ${outputFilename}")
    f => result.foreach{
      case (x, y, ccid) => f.println(s"${x}\t${y}\t${ccid}")
    }
  }
}
