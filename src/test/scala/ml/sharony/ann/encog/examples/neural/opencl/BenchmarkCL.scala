package ml.sharony.ann.encog.examples.neural.opencl

import org.encog.Encog
import org.encog.ml.data.basic.BasicMLDataSet
import org.encog.neural.networks.BasicNetwork
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation
import org.encog.util.benchmark.RandomTrainingFactory
import org.encog.util.simple.EncogUtility
import org.encog.util.{Format, Stopwatch}

/**
  * encog-opencl-test
  *
  * <a href="https://github.com/leadtune/encog-java/blob/master/encog-examples/src/org/encog/examples/neural/opencl/BenchmarkCL.java">LeadTune's encog-java example: BenchmarkCL</a>:
  *
  * "Performs a simple benchmark of your first OpenCL device, compared to the CPU.
  * If you have multiple OpenCL devices(i.e. two GPU's) this benchmark will only
  * take advantage of one. To see multiple OpenCL devices used in parallel, use
  * the BenchmarkConcurrent example."
  *
  * However, there is no longer an Encog.getInstance().getCL() method, and no way to directly access the OpenCL devices.
  * Edited by Inon Sharony on 2017-07-09.
  *
  */
object BenchmarkCL {
  // / <summary>
  // / Program entry point.
  // / </summary>
  // / <param name="args">Not used.</param>
  def main(args: Array[String]) {
    try {
      val outputSize = 2
      val inputSize = 10
      val trainingSize = 100000
      val training = RandomTrainingFactory.generate(1000, trainingSize, inputSize, outputSize, -1, 1)
      val network = EncogUtility.simpleFeedForward(training.getInputSize, 6, 2, training.getIdealSize, true)
      network.reset()
      System.out.println("Running OpenCL test.")
      val clTime = benchmarkCL(network, training)
      System.out.println("OpenCL test took " + clTime + "ms.")
      System.out.println()
      System.out.println("Running non-OpenCL test.")
      val cpuTime = benchmarkCPU(network, training)
      System.out.println("Non-OpenCL test took " + cpuTime + "ms.")
      System.out.println()
      val percent = Format.formatPercent(cpuTime.toDouble / clTime.toDouble)
      System.out.println("OpenCL Performed at " + percent + " the speed of non-OpenCL")
    } catch {
      case ex: Exception =>
        System.out.println("Can't startup CL, make sure you have drivers loaded.")
        System.out.println(ex.toString)
    } finally Encog.getInstance.shutdown()
  }

  def benchmarkCPU(network: BasicNetwork, training: BasicMLDataSet): Long = {
    val train = new ResilientPropagation(network, training)
    train.iteration() // warm-up

    val stopwatch = new Stopwatch
    stopwatch.start()
    var i = 0
    while ( {i < 100}) {
      train.iteration()

      {i += 1; i - 1}
    }
    stopwatch.stop()
    stopwatch.getElapsedMilliseconds
  }

  def benchmarkCL(network: BasicNetwork, training: BasicMLDataSet): Long = {
    val train = new ResilientPropagation(network, training)
    train.iteration()
    val stopwatch = new Stopwatch
    stopwatch.start()
    var i = 0
    while ( {i < 100}) {
      train.iteration()

      {i += 1; i - 1}
    }
    stopwatch.stop()
    stopwatch.getElapsedMilliseconds
  }
}
