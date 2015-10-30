Explore DL4J
====

Example code to explore for using [Deeplearning4J](http://deeplearning4j.org/) in Scala.

## Compilation and setup

You need Java 7 or higher and `sbt` installed.

Commands to checkout the repository, compile the code and add the bin directory to your path.
  
```
$ mkdir devel
$ cd devel
$ git clone https://github.com/jasonbaldridge/explore-dl4j.git
$ cd explore-dl4j/
$ export PATH=$PATH:~/devel/explore-dl4j/bin/
$ sbt stage
```

Feel free to use a different directory to put the code in and add the PATH to your bash profile, etc.

Note: if you change the code, you'll need to run `sbt stage` for those changes be available to the `bin` scripts. (Use `compile` in sbt while developing as usual to catch and fix errors.)

### Help

Use the `--help` option with any of the commands in `explore-dl4j/bin` to get all options available.

### Shuffling

Add this to your `.bash_profile` if you don't have the `shuf` command on your system. 

```
$ alias shuf="perl -MList::Util -e 'print List::Util::shuffle <>'"
$ source ~/.bash_profile
```
This randomly reorders the lines in a text file. It is generally useful to have around for manipulating files. (I've found this method to be easier than installing actual a prebuilt 

## Instructions for simulated data

Go to the `explore-dl4j/simulation_data` dir and follow the instructions below.

### Linearly separable data

Generate the data.

```
$ R CMD BATCH generate_linear_data.R
```

Train and evaluate a one layer MLN, which should be sufficient to perform the task.

```
$ run-simple --train-file simulated_linear_data_train.csv --eval-file simulated_linear_data_eval.csv --num-layers 1 > out_linear_one_layer.txt 2>&1
```

You should see something like this.

```
$ tail out_linear_one_layer.txt
==========================Scores========================================
 Accuracy:  0.9999
 Precision: 1
 Recall:    0.9998
 F1 Score:  0.9998999899989999
===========================================================================
```

Looks good!


### Non-linearly separable data

Generate the data, which produces a four-dimensional ball surrounded by a four-dimensional ring. There is no hyperplane that can separate these two classes, so it makes deeper networks more interesting in this case.

```
$ R CMD BATCH generate_saturn_data.R
```

Verify that a one-layer network fails to separate the classes.

```
$ run-simple --train-file simulated_saturn_data_train.csv --eval-file simulated_saturn_data_eval.csv --num-layers 1 > out_saturn_one_layer.txt 2>&1
```

Failure, as expected.

```
$ tail 
==========================Scores========================================
 Accuracy:  0.4752
 Precision: 0.4792
 Recall:    0.4133
 F1 Score:  0.44383213225943197
===========================================================================
```

Now, try with two layers.

```
$ run-simple --train-file simulated_saturn_data_train.csv --eval-file simulated_saturn_data_eval.csv --num-layers 2 > out_saturn_two_layer.txt 2>&1
```

Not there yet with this:

```
==========================Scores========================================
 Accuracy:  0.5066
 Precision: 0.5066
 Recall:    1
 F1 Score:  0.6725076330811097
===========================================================================
```

Time to mess around with initialization and model structure, etc.

## Instructions for sentiment classifier
  
Download the [sentiment140 data](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip). Make a directory, say `~/data/sentiment140`, and put it there. Unzip the file. This gives you two files: `testdata.manual.2009.06.14.csv` and `training.1600000.processed.noemoticon.csv`.

We need to shuffle the data because it is ordered by category and many machine learning algorithms don't behave well with such data. (E.g. online gradient descent.)

```
$ shuf < training.1600000.processed.noemoticon.csv > shuffled_training.processed.noemoticon.csv
```

We first need to train the word2vec vectors.

```
$ train-word-vectors --train-file shuffled_training.processed.noemoticon.csv --output-file sentiment_word_vectors.txt --num-dimensions 200 --input-type sentiment140
```

Run a sentiment classifier experiment. We'll run with less data to start with to ensure it is working. Create a file with just 10k examples and provide that as the input to `run-sentiment`.

```
$ head -10000 shuffled_training.processed.noemoticon.csv > small_training.processed.noemoticon.csv
$ run-sentiment --train-file small_training.processed.noemoticon.csv --vector-file sentiment_word_vectors.txt --eval-file testdata.manual.2009.06.14.csv 
```

This should run, though it gets accuracy no better than chance. This is running a single layer network (logistic regression).

```
$ run-sentiment --train-file shuffled_training.processed.noemoticon.csv --vector-file sentiment_word_vectors.txt --eval-file testdata.manual.2009.06.14.csv
```

