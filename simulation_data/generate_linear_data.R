### 10/29/2015
### Simulate a two-class linearly separable classification problem.
### Label 0 is the "negative" class.
### Label 1 is the "positive" class.
### Author: Jason Baldridge

# Create a matrix given a label, the class means of four dimensions,
# the number of items, and the standard deviation. Values are sampled
# normally according to the mean and stdev for each column.
create_matrix = function(label, mu, n, dev=.1) {
  d = length(mu)
  x = t(matrix(rnorm(n*d, mu, dev), ncol=n))
  cbind(rep(label,n),x)
 }

# Num input dimensions (the "features").
numDimensions = 10
  
# Sample the means for four dimensions for a positive class.
pos = runif(numDimensions,min=-1,max=1)
  
# Sample the means for four dimensions for a negative class.
neg = runif(numDimensions,min=-1,max=1)

# Create training data.
numTraining = 100000/2
training_data = as.matrix(rbind(create_matrix(1,pos,numTraining),create_matrix(0,neg,numTraining)))
shuffled_training_data = training_data[sample(nrow(training_data)),]
write.table(shuffled_training_data,file="simulated_linear_data_train.csv",row.names=FALSE,col.names=FALSE,quote=FALSE,sep=",")

# Create eval data. Make the stdev bigger to make it a bit more interesting..
numEval = 10000/2
evalDev = .3
eval_data = as.matrix(rbind(create_matrix(1,pos,numEval,evalDev),create_matrix(0,neg,numEval,evalDev)))
shuffled_eval_data = eval_data[sample(nrow(eval_data)),]
write.table(shuffled_eval_data,file="simulated_linear_data_eval.csv",row.names=FALSE,col.names=FALSE,quote=FALSE,sep=",")

