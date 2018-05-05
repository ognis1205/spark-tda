/*
 * Collection of Functions
 * 
 * USAGE:
 *   - To load this file from spark-shell simply do
 *     spark> :load PATH_TO_THIS_FILE
 */
import java.io.{File, PrintWriter}
import org.apache.spark.graphx.{Graph, EdgeRDD, VertexRDD}

def time[R](block: => R): R = {  
    val s = System.currentTimeMillis
    val result = block
    val e = System.currentTimeMillis
    println(s"ELAPSED TIME: ${(e - s) / 1000.0} sec")
    result
}
