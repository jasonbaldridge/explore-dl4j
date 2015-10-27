Explore DL4J
====

Example code to explore for using DL4J in Scala.

## Compilation and setup

Commands to checkout the repository, compile the code and add the bin directory to your path.
  
```
$ mkdir devel
$ cd devel
$ git clone https://github.com/jasonbaldridge/explore-dl4j.git
$ sbt stage
$ export PATH=$PATH:~/devel/explore-dl4j/bin/
```

Feel free to use a different directory to put the code in and add the PATH to your bash profile, etc.

Also, add this to your `.bash_profile` if you don't have the `shuf` command on your system.

```
$ alias shuf="perl -MList::Util -e 'print List::Util::shuffle <>'"`
$ source ~/.bash_profile
```

## Instructions for sentiment classifier
  
Download the [sentiment140 data](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip). Make a directory, say `~/data/sentiment140`, and put it there. Unzip the file. This gives you two files: `testdata.manual.2009.06.14.csv` and `training.1600000.processed.noemoticon.csv`.

We need to shuffle the data because it is ordered by category and many machine learning algorithms don't behave well with such data. (E.g. online gradient descent.)

```
$ shuf < training.1600000.processed.noemoticon.csv > shuffled_training.processed.noemoticon.csv
```

We first need to train the word2vec vectors.

```
$ train-word-vectors -trainfile shuffled_training.processed.noemoticon.csv --vectorfile sentiment_word_vectors.txt --vectorlength 200
```

Run a sentiment classifier experiment. We'll run with less data to start with to ensure it is working, which is specified with the `--maxtraining`` argument.

```
$ run-sentiment --trainfile shuffled_training.processed.noemoticon.csv --vectorfile sentiment_word_vectors.txt --evalfile testdata.manual.2009.06.14.csv
```

