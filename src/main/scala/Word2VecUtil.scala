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
import org.nd4j.linalg.ops.transforms.Transforms.exp
import org.slf4j.LoggerFactory
import java.io.File


class Word2VecUtil(wordVectors: WordVectors) {

  // How long are the vectors?
  val vectorLength = wordVectors.lookupTable.vectors.next.length

  def vectorizeDocument(words: Array[String]) = {
    val sumVector = Nd4j.zeros(1, vectorLength)
    var wordsFound = 0
    for (word <- words; if wordVectors.hasWord(word)) {
      wordsFound += 1
      sumVector.addi(wordVectors.getWordVectorMatrix(word))
    }
    sumVector.div(wordsFound)
  }

}


/**
  * Static methods for training and using w2v vectors.
  */
object Word2VecUtil {

  lazy val log = LoggerFactory.getLogger(expdl4j.Word2VecUtil.getClass)

  /**
    * Main method for training vectors. Accessible via the
    * bin/train-w2v script.
    */
  def main(args: Array[String]) {

    val conf = new Word2VecCommand(args)
    conf.afterInit()

    val trainFileName = conf.trainfile()
    val wordVectorFileName = conf.vectorfile()
    val vectorLength = conf.vectorlength()

    if (new File(wordVectorFileName).isFile)
      println(s"File $wordVectorFileName already exists. Exiting.")
    else {
      trainAndSaveWord2Vec(trainFileName, wordVectorFileName, vectorLength)
    }

  }
  
  /**
    * Train word2vec model and save it to disk.
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
    wordVectors: WordVectors
  ) = {

    // How long are the vectors?
    val vectorLength = wordVectors.lookupTable.vectors.next.length
      
    // Accumulator for the featurized items (label + document as vector).
    val data = new ListBuffer[(INDArray,INDArray)]()

    val w2vUtil = new Word2VecUtil(wordVectors)
    
    // Parse the csv file again to get label and average word vector
    val it = new Sentiment140Iterator(inputFilename)
    while (it.hasNext()) {

      // The iterator returns an option. This enables a clean solution
      // to skipping neutral items for this particular task.
      it.nextLabelAndSentence.map { case(label,words) =>
        data.append((Nd4j.create(label),w2vUtil.vectorizeDocument(words)))
      }
    }
    
    data.toList
  }


}
