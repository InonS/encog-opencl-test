package ml.sharony.ann.encog.examples.neural.opencl

import java.lang.Math.{max, min}

import org.encog.ml.data.basic.BasicMLDataSet
import org.encog.neural.networks.BasicNetwork
import org.encog.neural.networks.training.concurrent.ConcurrentTrainingManager
import org.encog.neural.networks.training.concurrent.jobs.RPROPJob
import org.encog.util.benchmark.RandomTrainingFactory
import org.encog.util.logging.EncogLogging
import org.encog.util.simple.EncogUtility

/**
  * encog-opencl-test
  * Created by Inon Sharony on 2017-07-09.
  */
object EncogOpenCLTest extends App {
  test()

  private def test() {
    val trainMgr = ConcurrentTrainingManager.getInstance()
    trainMgr.detectPerformers(true)
    EncogLogging.log(EncogLogging.LEVEL_DEBUG, trainMgr.toString)
    val training = generateDataset()
    val network = constructNetwork(training)
    val trainingJob = new RPROPJob(network, training, false)
    /*// As seen in Backpropagation (otherwise see Cross-Validation strategy in [EncogModel]
    trainingJob.addStrategy(new SmartLearningRate)
    trainingJob.addStrategy(new SmartMomentum)
    trainingJob.createTrainer(false)*/
    trainMgr.addTrainingJob(trainingJob)
    trainMgr.run()
    trainMgr.start()
    trainMgr.join()
  }

  private def generateDataset(outputSize: Int = 2, inputSize: Int = 10, trainingSize: Int = 1e5.toInt, range: Tuple2[Double, Double] = (-1, 1), seed: Long = 2017): BasicMLDataSet = {
    RandomTrainingFactory.generate(seed, trainingSize, inputSize, outputSize, range._1, range._2)
  }

  private def constructNetwork(dataset: BasicMLDataSet): BasicNetwork = {
    val hidden1Nodes = min(6, dataset.getInputSize)
    val hidden2Nodes = max(2, dataset.getIdealSize)
    val tanh = true
    val network = EncogUtility.simpleFeedForward(dataset.getInputSize, hidden1Nodes, hidden2Nodes, dataset.getIdealSize, tanh)
    network.reset()
    network
  }
}
