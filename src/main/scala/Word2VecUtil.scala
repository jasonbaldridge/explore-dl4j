package expdl4j

import org.deeplearning4j.models.embeddings.WeightLookupTable
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache
import scala.collection.mutable.ListBuffer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory
import java.io.File

/**
  * Static methods for training and using w2v vectors.
  */
object Word2VecUtil {

  lazy val log = LoggerFactory.getLogger(expdl4j.Word2VecUtil.getClass)

  /**
    * Train word2vec model, save it to disk and return it.
    */
  def trainAndSaveWord2Vec(
    trainFileName: String,
    word2vecTxtFilePath: String,
    vectorLength: Int = 200
  ) {
    val it = new Sentiment140Iterator(trainFileName)
    val cache = new InMemoryLookupCache()
    val table = new InMemoryLookupTable.Builder()
      .vectorLength(vectorLength)
      .useAdaGrad(false)
      .cache(cache)
      .lr(0.025f).build()
    
    log.info("Building model....")
    val vec = new Word2Vec.Builder()
      .minWordFrequency(5).iterations(3)
      .layerSize(vectorLength).lookupTable(table)
      .vocabCache(cache).seed(42)
      .windowSize(5).iterate(it).build()
    
    log.info("Training model...")
    vec.fit()
    
    log.info("Writing word vectors to file...")
    WordVectorSerializer.writeWordVectors(vec, word2vecTxtFilePath)
  }

  /**
    * Load vectors from disk.
    */
  def getVectors(word2vecTxtFilePath: String) = 
    WordVectorSerializer.loadTxtVectors(new File(word2vecTxtFilePath))

  /**
    * Given input documents and word vectors, compute vec-length representation of each
    * document for input to net.
    */
  def computeAvgWordVector(
    inputFilename: String,
    wordVectors: WordVectors,
    vectorLength: Int = 200
  ) = {

    // Accumulator for the featurized items (label + document as vector).
    val data = new ListBuffer[(INDArray,INDArray)]()
    
    // Parse the csv file again to get label and average word vector
    val it = new Sentiment140Iterator(inputFilename)
    while (it.hasNext()) {

      // The iterator returns an option. This enables a clean solution to skipping neutral
      // items for this particular task.
      val infoOpt = it.nextLabelAndSentence
      infoOpt.map { info =>
        
        val (label,words) = info

        // Add label to list
        val labelVector = Nd4j.create(label)
        
        // Add avg tweet vector to list
        val sumTweetVector = Nd4j.zeros(1, vectorLength)
        words.foreach { word =>
          val vec = wordVectors.getWordVectorMatrix(word)
          sumTweetVector.addi(wordVectors.getWordVectorMatrix(word))
        }
        val averageTweetVector = sumTweetVector.div(words.length)
        data.append((labelVector,averageTweetVector))
      }
    }
    
    data.toList
  }


}
