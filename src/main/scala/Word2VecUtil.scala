package expdl4j

import org.deeplearning4j.models.embeddings.WeightLookupTable
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache
import org.deeplearning4j.text.sentenceiterator.BaseSentenceIterator
import scala.collection.mutable.ListBuffer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms.exp
import org.slf4j.LoggerFactory
import java.io.File


class Word2VecUtil(wordVectors: WordVectors) {

  // Lookup the number of dimensions.
  lazy val numDimensions = wordVectors.lookupTable.vectors.next.length

  /**
    * Given an array of tokens, compute the average vector for all
    * words in the word vector model.
    */
  def vectorizeDocument(words: Array[String]) = {
    val sumVector = Nd4j.zeros(1, numDimensions)
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

    val trainFileName = conf.trainFile()
    val wordVectorFileName = conf.outputFile()
    val vectorLength = conf.numDimensions()

    if (new File(wordVectorFileName).isFile) {
      println(s"File $wordVectorFileName already exists. Exiting.")
    } else {
      val sentenceIterator = conf.inputType() match {
        case "sentiment140" => new Sentiment140Iterator(trainFileName)
        case _ => new RawSentenceIterator(trainFileName) // Default
      }
      trainAndSaveWord2Vec(sentenceIterator, wordVectorFileName, vectorLength)
    }

  }
  
  /**
    * Train word2vec model and save it to disk.
    */
  def trainAndSaveWord2Vec(
    sentenceIterator: BaseSentenceIterator,
    word2vecTxtFilePath: String,
    vectorLength: Int = 200
  ) {

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
      .windowSize(5).iterate(sentenceIterator).build()
    
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

}
