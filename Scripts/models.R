rm(list = ls())

setwd( "\\\\prism.nas.gatech.edu/hreddivari3/vlab/documents/cdaproject/facebook3")
#install.packages(c("data.table","e1071","nnet","randomForest","pROC","adabag"))
#######################################################################
#Models

load("extractedFeaturesinit3.Rda")

myData <- train
# Set-up response and predictor variables
myData <- myData[,-1]
myResponse=as.numeric(myData$outcome)
myPredictors=data.matrix(myData[,!(attributes(myData)$names %in% c("outcome","crime_median_p","crime_mean_pw","crime_mean_t","crime_mean_tw"))])
mymaintest <- test[,-1]
mymaintestX=data.matrix(test[,!(attributes(test)$names %in% c("bidder_id","outcome","crime_median_p","crime_mean_pw","crime_mean_t","crime_mean_tw"))])

# make all data mean zero and variance one
myPredictors_mean = apply(myPredictors,2,function(xx){
  return(mean(xx,na.rm=TRUE))
})
myPredictors_sd = apply(myPredictors,2,function(xx){
  return(sd(xx,na.rm=TRUE))
})
myX=apply(myPredictors,2,function(xx){
  return((xx-mean(xx,na.rm=TRUE))/sd(xx,na.rm=TRUE))
})
myY = myResponse

############# Develop Prediction Models ###################
# creat a taining set and verification set
set.seed(30126)
nFolds=10
myIndices=sample(length(myResponse),ceiling(length(myResponse)/nFolds))
trainingX = myX[-myIndices,]
trainingY = myY[-myIndices]
testX = myX[myIndices,]
testY = myY[myIndices]
myYFactor = factor(myY)
trainingYFactor = myYFactor[-myIndices]
testYFactor = myYFactor[myIndices]


# Support vector machine ~ cost
library(e1071)
svm.model = svm(trainingX, trainingY, type = "C", cost = 10)
svm.pred <- predict(svm.model, testX)
svm.prob = predict(svm.model, testX, probability = TRUE)
svmt=table(pred = svm.pred, true = testY)
svmt

# neural network ~ size ~ decay
library(nnet)
lambda = 0.0001
fit=nnet(trainingYFactor~.,data=trainingX,weights=rep(1,length(trainingYFactor)),size=10,
         decay=lambda,MaxNWts=10000,maxit=250)
NNPred=predict(fit,newdata=testX, type = "class")
NNProb=predict(fit,newdata=testX, type = "raw")
NNt=table(pred = NNPred, true = testY)
NNt

# random forest ~ ntree
library(randomForest)
RFmodel = randomForest(x = trainingX, y = trainingYFactor, ntree = 10)
RFPred = predict(RFmodel, testX, type = "class")
RFProb = predict(RFmodel, testX, type = "prob")[,2]
RFt=table(pred = RFPred, true = testY)
RFt

# logistic regression ~ k ~ direction
LRdata = data.frame(trainingX, trainingYFactor)
LRtestdata = data.frame(testX, testYFactor)
LRmodel = glm( trainingYFactor ~ . , family = binomial(logit),data = LRdata)
LRmodelstep = step(LRmodel,k=2,direction = "both")
LRProb = predict(LRmodelstep, newdata = data.frame(testX), type = "response")
LRPred = ifelse(predict(LRmodel, newdata = data.frame(testX), type = "response")>.5,1,0)
LRProb = predict(LRmodel, newdata = data.frame(testX), type = "response")
LRt=table(pred = LRPred, true = testY)
LRt

# boosted model ~ mfinal ~ maxdepth 
library(adabag)
myYfactor = as.factor(myY)
adadata = data.frame(myYfactor,myX) 
adaTrain = adadata[-myIndices,] 
adaTest = adadata[myIndices,]
myM = 100
adamodel = boosting(myYfactor~.,data=adaTrain,mfinal=myM,coeflearn="Freund",
                    control=rpart.control(maxdepth=10)) 
adaPred = predict(adamodel, newdata=adaTest) 
adaProb = adaPred$prob[,2]
adat=table(pred = as.numeric(adaPred$class), true = testY) 
adat


svmt
NNt
RFt
LRt
adat

## 10 folds validation
library(pROC)
# Support vector machine ~ cost
library(e1071)
set.seed(123456789)
nFolds = 10
table = matrix(0,2,2)
for (jj in 1:nFolds){
  # Generate training and testing responses and predictors for each fold
  myIndices=sample(length(myY))[1:round(length(myY)/nFolds)]
  myXtrain = myX[-myIndices,]
  myYtrain = myY[-myIndices]
  myXtest = myX[myIndices,]
  myYtest = myY[myIndices]
  # Set tuning parameters
  myC = 10
  # Calculate the SVM model
  svm.model = svm(myXtrain, myYtrain, type = "C", cost = myC,probability = TRUE)
  # Set threshold
  threshold = 0.5
  svm.prob = predict(svm.model, myXtest, probability = TRUE)
  svm.prob
  svm.pred = ifelse(attr(svm.prob,"probabilities")[,colnames(attr(svm.prob,"probabilities"))==1]>threshold,1,0)
  svmt = table(pred = svm.pred, true = myYtest)
  table = table + svmt
  myRoc <- roc(response = myYtest, predictor = attr(svm.prob,"probabilities")[,colnames(attr(svm.prob,"probabilities"))==1], auc.polygon=TRUE, grid=TRUE, plot=FALSE)
  plot(myRoc)
}
######################### SVM tuning ############################
## SVM ~ cost
library(e1071)
set.seed(123456789)
nFolds = 10
table = matrix(0,2,2)
for (jj in 1:nFolds){
  # Generate training and testing responses and predictors for each fold
  myIndices=sample(length(myY))[1:round(length(myY)/nFolds)]
  myXtrain = myX[-myIndices,]
  myYtrain = myY[-myIndices]
  myXtest = myX[myIndices,]
  myYtest = myY[myIndices]
  # Set tuning parameters
  myC = 10
  # Calculate the SVM model
  svm.model = svm(myXtrain, myYtrain, type = "C", cost = myC,probability = TRUE)
  # Set threshold
  threshold = 0.5
  svm.prob = predict(svm.model, myXtest, probability = TRUE)
  svm.pred = ifelse(attr(svm.prob,"probabilities")[,colnames(attr(svm.prob,"probabilities"))==1]>threshold,1,0)
  svmt = table(pred = svm.pred, true = myYtest)
  print(svmt)
  table = table + svmt
  myRoc <- roc(response = myYtest, predictor = attr(svm.prob,"probabilities")[,colnames(attr(svm.prob,"probabilities"))==1], auc.polygon=TRUE, grid=TRUE, plot=FALSE)
  plot(myRoc)
}

## SVM ~ cost ~ gamma
library(e1071)
set.seed(123456789)
nFolds = 10
iter = 10
myCs = c(0.001,0.005,0.02,0.05,0.08,0.1)
#cost -- cost of constraints violation (default: 1)—it is the ‘C’-constant of the regularization term in the Lagrange formulation
#gammas -- parameter needed for all kernels except linear (default: 1/(data dimension))
myGammas = c(0.01,0.05,0.1,1)
tuningProb = array(NA,dim = c(iter,length(myCs),length(myGammas),round(length(myY)/nFolds)))
tuningRes = array(NA,dim = c(iter,round(length(myY)/nFolds)))
for (jj in 1:iter){
  # Generate training and testing responses and predictors for each fold
  myIndices=sample(length(myY))[1:round(length(myY)/nFolds)]
  myXtrain = myX[-myIndices,]
  myYtrain = myY[-myIndices]
  myXtest = myX[myIndices,]
  myYtest = myY[myIndices]
  for (ii in 1:length(myCs)){
    # Set tuning parameters
    myC = myCs[ii]
    for (kk in 1:length(myGammas)){
      myGamma = myGammas[kk]
      # Calculate the SVM model
      svm.model = svm(myXtrain, myYtrain, type = "C", gamma = myGamma, cost = myC,probability = TRUE)
      # Predict probability
      svm.prob = predict(svm.model, myXtest, probability = TRUE)
      tuningProb[jj,ii,kk,] = attr(svm.prob,"probabilities")[,colnames(attr(svm.prob,"probabilities"))==1]
    }
  }
  tuningRes[jj,] = myYtest
}
# compute ROC
roc_res = array(NA,dim = c(iter*round(length(myY)/nFolds),1))
roc_prob = array(NA,dim = c(iter*round(length(myY)/nFolds),length(myCs),length(myGammas)))
for (jj in 1:iter){
  roc_res[((jj-1)*round(length(myY)/nFolds)+1):(jj*round(length(myY)/nFolds))] = tuningRes[jj,]
  for (ii in 1:length(myCs)){
    for (kk in 1:length(myGammas)){
      roc_prob[((jj-1)*round(length(myY)/nFolds)+1):(jj*round(length(myY)/nFolds)),ii,kk] = tuningProb[jj,ii,kk,]
    }
  }
}
# compute ROC and plot ROC vs. tuning parameters
myRocList = list()
tuningAUC = matrix(NA,length(myCs),length(myGammas))
for (ii in 1:length(myCs)){
  for (kk in 1:length(myGammas)){
    myRocList[[(ii-1)*length(myGammas)+kk]] = roc(response = roc_res, predictor = roc_prob[,ii,kk], auc.polygon=TRUE, grid=TRUE, plot=TRUE)
    tuningAUC[ii,kk] = myRocList[[(ii-1)*length(myGammas)+kk]]$auc
  }
}
library(corrplot)
rownames(tuningAUC) = c("cost=0.001","cost=0.005","cost=0.02","cost=0.05","cost=0.08","cost=0.1")
colnames(tuningAUC) = c("gamma=0.01","gamma=0.05","gamma=0.1","gamma=1")
corrplot(t(tuningAUC), method = "ellipse", order = "original", is.corr = FALSE,col = colorRampPalette(c("green","navyblue"))(100))
######################################### Neural Network tuning #######################################
## Neural Network ~ size ~ decay
library(nnet)
set.seed(123456789)
nFolds = 10
iter = 10
mySizes = c(2,4,5,10,15)
myDecays = c(0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01)
# size number of units in the hidden layer. Can be zero if there are skip-layer units.
# decay parameter for weight decay. Default 0
tuningProb = array(NA,dim = c(iter,length(mySizes),length(myDecays),round(length(myY)/nFolds)))
tuningRes = array(NA,dim = c(iter,round(length(myY)/nFolds)))
for (jj in 1:iter){
  # Generate training and testing responses and predictors for each fold
  myIndices=sample(length(myY))[1:round(length(myY)/nFolds)]
  myXtrain = myX[-myIndices,]
  myYtrain = myY[-myIndices]
  myYtrainFactor = as.factor(myYtrain)
  myXtest = myX[myIndices,]
  myYtest = myY[myIndices]
  for (ii in 1:length(mySizes)){
    # Set tuning parameters
    mySize = mySizes[ii]
    for (kk in 1:length(myDecays)){
      myDecay = myDecays[kk]
      # Calculate the NN model
      fit=nnet(myYtrainFactor~.,data=myXtrain,weights=rep(1,length(myYtrainFactor)),size=mySize,
               decay=myDecay,MaxNWts=10000,maxit=10000,trace=FALSE)
      # Predict probability
      NNProb=predict(fit,newdata=myXtest, type = "raw")
      tuningProb[jj,ii,kk,] = NNProb
    }
  }
  tuningRes[jj,] = myYtest
}
# compute ROC
roc_res = array(NA,dim = c(iter*round(length(myY)/nFolds),1))
roc_prob = array(NA,dim = c(iter*round(length(myY)/nFolds),length(mySizes),length(myDecays)))
for (jj in 1:iter){
  roc_res[((jj-1)*round(length(myY)/nFolds)+1):(jj*round(length(myY)/nFolds))] = tuningRes[jj,]
  for (ii in 1:length(mySizes)){
    for (kk in 1:length(myDecays)){
      roc_prob[((jj-1)*round(length(myY)/nFolds)+1):(jj*round(length(myY)/nFolds)),ii,kk] = tuningProb[jj,ii,kk,]
    }
  }
}
# compute ROC and plot ROC vs. tuning parameters
myRocList = list()
tuningAUC = matrix(NA,length(mySizes),length(myDecays))
for (ii in 1:length(mySizes)){
  for (kk in 1:length(myDecays)){
    myRocList[[(ii-1)*length(myDecays)+kk]] = roc(response = roc_res, predictor = roc_prob[,ii,kk], auc.polygon=TRUE, grid=TRUE, plot=TRUE)
    tuningAUC[ii,kk] = myRocList[[(ii-1)*length(myDecays)+kk]]$auc
  }
}
plot(tuningAUC)
image(t(tuningAUC[nrow(tuningAUC):1,] ), axes=FALSE, zlim=c(-4,4), col=rainbow(21))
library(corrplot)
rownames(tuningAUC) = c("size=2","size=4","size=5","size=10","size=15")
colnames(tuningAUC) = c("decay=0.00001","decay=0.00005","decay=0.0001","decay=0.0005","decay=0.001","decay=0.005","decay=0.01")
corrplot(tuningAUC, method = "ellipse", order = "original", is.corr = FALSE,col = colorRampPalette(c("green","navyblue"))(100))
######################################### Random Forest tuning #######################################
## random forest ~ ntree
library(randomForest)
set.seed(123456789)
nFolds = 10
iter = 10
myNtrees = seq(10,150,10)
tuningProb = array(NA,dim = c(iter,length(myNtrees),round(length(myY)/nFolds)))
tuningRes = array(NA,dim = c(iter,round(length(myY)/nFolds)))
for (jj in 1:iter){
  # Generate training and testing responses and predictors for each fold
  myIndices=sample(length(myY))[1:round(length(myY)/nFolds)]
  myXtrain = myX[-myIndices,]
  myYtrain = myY[-myIndices]
  myYtrainFactor = as.factor(myYtrain)
  myXtest = myX[myIndices,]
  myYtest = as.factor(myY[myIndices])
  for (ii in 1:length(myNtrees)){
    # Set tuning parameters
    myNtree = myNtrees[ii]
    # Calculate the RF model
    RFmodel = randomForest(x = myXtrain, y = myYtrainFactor, ntree = myNtree)
    # Predict probability
    RFProb = predict(RFmodel, myXtest, type = "prob")[,2]
    tuningProb[jj,ii,] = RFProb
  }
  tuningRes[jj,] = myYtest
}
# compute ROC
roc_res = array(NA,dim = c(iter*round(length(myY)/nFolds),1))
roc_prob = array(NA,dim = c(iter*round(length(myY)/nFolds),length(myNtrees)))
for (jj in 1:iter){
  roc_res[((jj-1)*round(length(myY)/nFolds)+1):(jj*round(length(myY)/nFolds))] = tuningRes[jj,]
  for (ii in 1:length(myNtrees)){
    roc_prob[((jj-1)*round(length(myY)/nFolds)+1):(jj*round(length(myY)/nFolds)),ii] = tuningProb[jj,ii,]
  }
}
# compute ROC and plot ROC vs. tuning parameters
myRocList = list()
tuningAUC = matrix(NA,length(myNtrees),1)
for (ii in 1:length(myNtrees)){
  myRocList[[ii]] = roc(response = roc_res, predictor = roc_prob[,ii], auc.polygon=TRUE, grid=TRUE, plot=TRUE)
  tuningAUC[ii,1] = myRocList[[ii]]$auc
}
plot(myNtrees,tuningAUC)


######################################### Logistic Regression tuning #######################################
# logistic regression ~ k ~ direction 
library(adabag) 
set.seed(123456789)
nFolds = 10
iter = 10
myDirections = c("forward","backward","both")
myKs = c(1,2,3,4,5,6,7,8,9)
tuningProb = array(NA,dim = c(iter,length(myDirections),length(myKs),round(length(myY)/nFolds)))
tuningRes = array(NA,dim = c(iter,round(length(myY)/nFolds)))
for (jj in 1:iter){
  # Generate training and testing responses and predictors for each fold
  myIndices=sample(length(myY))[1:round(length(myY)/nFolds)]
  myYfactor = as.factor(myY) 
  myXtrain = myX[-myIndices,] 
  myYtrain = myYfactor[-myIndices] 
  myXtest = myX[myIndices,] 
  myYtest = myYfactor[myIndices]
  LRtrain = data.frame(myXtrain, myYtrain)
  for (ii in 1:length(myDirections)){ # Set tuning parameters 
    myDirection = myDirections[ii] 
    for (kk in 1:length(myKs)){
      myK = myKs[kk]
      # Calculate the LR model
      LRmodel = glm( myYtrain ~., family = binomial(logit),data = LRtrain)
      LRmodelstep = step(LRmodel,k=myK,direction = myDirection)
      # Predict probability
      LRProb = predict(LRmodelstep, newdata = data.frame(myXtest), type = "response")
      tuningProb[jj,ii,kk,] = LRProb 
    }
  }
  tuningRes[jj,] = myYtest
}
# compute ROC
roc_res = array(NA,dim = c(iter*round(length(myY)/nFolds),1))
roc_prob = array(NA,dim = c(iter*round(length(myY)/nFolds),length(myDirections),length(myKs)))
for (jj in 1:iter){
  roc_res[((jj- 1)*round(length(myY)/nFolds)+1):(jj*round(length(myY)/nFolds))] = tuningRes[jj,]
  for (ii in 1:length(myDirections)){ 
    for (kk in 1:length(myKs)){
      roc_prob[((jj-1)*round(length(myY)/nFolds)+1):(jj*round(length(myY)/nFolds)),ii,kk] = tuningProb[jj,ii,kk,]
    }
  }
}
# compute ROC and plot ROC vs. tuning parameters 
myRocList = list()
tuningAUC = matrix(NA,length(myDirections),length(myKs)) 
for (ii in 1:length(myDirections)){
  for (kk in 1:length(myKs)){ 
    myRocList[[(ii-1)*length(myKs)+kk]] = roc(response = as.factor(roc_res), predictor = roc_prob[,ii,kk], auc.polygon=TRUE, grid=TRUE, plot=TRUE)
    tuningAUC[ii,kk] = myRocList[[(ii- 1)*length(myKs)+kk]]$auc
  }
}
plot(tuningAUC)
image(t(tuningAUC[nrow(tuningAUC):1,] ), axes=FALSE, zlim=c(-4,4), col=rainbow(21))
library(corrplot)
rownames(tuningAUC) = c("forward","backward","both") 
colnames(tuningAUC) = c("k=1","k=2","k=3","k=4","k=5","k=6","k=7","k=8","k=9") 
corrplot(tuningAUC, method = "ellipse", order = "original", is.corr = FALSE,col = colorRampPalette(c("green","navyblue"))(100))


######################################### Boosting Tuning #######################################
# boosted model ~ mfinal ~ maxdepth 
library(adabag)
set.seed(123456789)
nFolds = 10
iter = 6
myMs = c(50,100,150)
myMaxdepths = c(2,4,6,8,10)
tuningProb = array(NA,dim = c(iter,length(myMs),length(myMaxdepths),round(length( myY)/nFolds)))
tuningRes = array(NA,dim = c(iter,round(length(myY)/nFolds)))
for (jj in 1:iter){
  # Generate training and testing responses and predictors for each fold
  myIndices=sample(length(myY))[1:round(length(myY)/nFolds)]
  myYfactor = as.factor(myY)
  adadata = data.frame(myYfactor,myX) 
  adaTrain = adadata[-myIndices,] 
  adaTest = adadata[myIndices,]
  for (ii in 1:length(myMs)){ 
    # Set tuning parameters 
    myM = myMs[ii]
    for (kk in 1:length(myMaxdepths)){ 
      print(kk)
      myMaxdepth = myMaxdepths[kk] 
      # Calculate the ADAmodel
      adamodel = boosting(myYfactor~.,data=adaTrain,mfinal=myM,coeflearn="Freund", control=rpart.control(maxdepth=myMaxdepth))
      # Predict probability
      adaPred = predict(adamodel, newdata=adaTest) 
      adaProb = adaPred$prob[,2]
      tuningProb[jj,ii,kk,] = adaProb 
    }
  }
  tuningRes[jj,] = adaTest[,1]
}
# compute ROC
roc_res = array(NA,dim = c(iter*round(length(myY)/nFolds),1))
roc_prob = array(NA,dim = c(iter*round(length(myY)/nFolds),length(myMs),length(myMaxdepths)))
for (jj in 1:iter){
  roc_res[((jj- 1)*round(length(myY)/nFolds)+1):(jj*round(length(myY)/nFolds))] = tuningRes[jj,]
  for (ii in 1:length(myMs)){
    for (kk in 1:length(myMaxdepths)){
      roc_prob[((jj- 1)*round(length(myY)/nFolds)+1):(jj*round(length(myY)/nFolds)),ii,kk] = tuningProb[jj,ii,kk,]
    }
  }
}

# compute ROC and plot ROC vs. tuning parameters 
myRocList = list()
tuningAUC = matrix(NA,length(myMs),length(myMaxdepths)) 
for (ii in 1:length(myMs)){
  for (kk in 1:length(myMaxdepths)){ 
    myRocList[[(ii-1)*length(myMaxdepths)+kk]] = roc(response = as.factor(roc_res), predictor = roc_prob[,ii,kk], auc.polygon=TRUE, grid=TRUE, plot=TRUE)
    tuningAUC[ii,kk] = myRocList[[(ii- 1)*length(myMaxdepths)+kk]]$auc
  }
}

plot(tuningAUC)
image(t(tuningAUC[nrow(tuningAUC):1,] ), axes=FALSE, zlim=c(-4,4), col=rainbow(21))
library(corrplot)
rownames(tuningAUC) = c("m=50","m=100","m=150") 
colnames(tuningAUC) = c("maxdepth=2","maxdepth=4","maxdepth=6","maxdepth =8","maxdepth=10")
corrplot(tuningAUC, method = "ellipse", order = "original", is.corr = FALSE,col = colorRampPalette(c("green","navyblue"))(100))

################################ calculate the cross- validation results using five tuned models###########################
nFolds = 10
niter = 10
foldLength = round(length(myY)/nFolds) 
yMatrix = matrix(NA, foldLength*nFolds,1) 
miuMatrix = matrix(NA, foldLength*nFolds,5)
library(e1071) 
library(nnet) 
library(randomForest) 
library(adabag) 
for (kk in 1:niter){ 
  myIndices=sample(length(myY))[1:foldLength] 
  myXtrain = myX[-myIndices,]
  myYtrain = myY[-myIndices]
  myXtest = myX[myIndices,]
  myYtest = myY[myIndices]
  # yMatrix
  yMatrix[((kk-1)*foldLength+1):(kk*foldLength)] = myYtest
  
  ################ miuMatrix ###########
  # SVM
  svm.model = svm(myXtrain, myYtrain, type = "C", gamma = 0.02, cost = 0.05,probability = TRUE)
  svm.prob = predict(svm.model, myXtest, probability = TRUE)
  miuMatrix[((kk-1)*foldLength+1):(kk*foldLength),1] = attr(svm.prob,"probabilities")[,colnames(attr(svm.prob,"probabilities"))==1]
  # Neural Network
  myYtrainFactor = as.factor(myYtrain)
  fit=nnet(myYtrainFactor~.,data=myXtrain,weights=rep(1,length(myYtrainFactor)),size=4,
           decay=0.01,MaxNWts=10000,maxit=10000,trace=FALSE) 
  NNProb=predict(fit,newdata=myXtest, type = "raw") 
  miuMatrix[((kk-1)*foldLength+1):(kk*foldLength),2] = NNProb
  # random forest
  RFmodel = randomForest(x = myXtrain, y = myYtrainFactor, ntree = 15)
  RFProb = predict(RFmodel, myXtest, type = "prob")[,2]
  miuMatrix[((kk-1)*foldLength+1):(kk*foldLength),3] = RFProb
  # boosted model
  myYfactor = as.factor(myY)
  adadata = data.frame(myYfactor,myX) 
  adaTrain = adadata[-myIndices,] 
  adaTest = adadata[myIndices,] 
  adamodel = boosting(myYfactor~.,data=adaTrain,mfinal=50,coeflearn ="Freund", control=rpart.control(maxdepth=10))
  adaPred = predict(adamodel, newdata=adaTest) 
  adaProb = adaPred$prob[,2] 
  miuMatrix[((kk-1)*foldLength+1):(kk*foldLength),4] = adaProb
  # Logistic Regression
  myYtrain = myYfactor[-myIndices]
  LRtrain = data.frame(myXtrain, myYtrain)
  LRmodel = glm(myYtrain~ ., family = binomial(logit),data = LRtrain)
  LRmodelstep = step(LRmodel,k=3,direction = "both")
  LRProb = predict(LRmodelstep, newdata = data.frame(myXtest), type = "response")
  miuMatrix[((kk-1)*foldLength+1):(kk*foldLength),5] = LRProb
}
D = 2*t(miuMatrix)%*%miuMatrix
d = 2*t(miuMatrix)%*%yMatrix
A = t(rbind(matrix(1,1,5),matrix(-1,1,5),diag(c(1,1,1,1,1)))) 
b = c(1, -1, 0, 0, 0, 0, 0)
library(quadprog)
solution = solve.QP(D,d,A,b,factorized = FALSE)
weight1 = solution$solution
weight2 = solution$unconstrained.solution

# SVM optimal params -- gamma 0.01 and cost 0.05
# Neural Net params -- size 15 and decay 0.01
# RF optimal trees - 60
# LR optimal - method both and k = 7
# boosting - m = 50 and maxdepth = 8
load("allmodels.RData")

svm2.model = svm(myX, myY, type = "C", cost = 0.05,gamma = 0.01, probability = TRUE)
svm2.pred <- predict(svm2.model, mymaintestX)
svm2.prob = predict(svm2.model, mymaintestX, probability = TRUE)

getMissingBidders <- function(table) {
    # Get missing bidders from the selected table
    db <- SQLite()
    dbconn <- dbConnect(drv = db, fr4db)
    query <- paste0(sprintf("SELECT DISTINCT bidder_id FROM %s ", table),
                    "WHERE bidder_id NOT IN ",
                    "(SELECT DISTINCT bidder_id FROM bids)")
    res <- dbGetQuery(dbconn, statement = query)
    dbDisconnect(dbconn)
    return(res$bidder_id)
}
missingTest <- getMissingBidders("test")
resultSVM <- rbind(resultSVM, cbind( bidder_id = missingTest, prediction = 0))
# Save result
write.csv(result, file = "submissionSVM.csv", quote = FALSE, row.names = FALSE)

library(nnet)
lambda = 0.0001
fit2=nnet(myYFactor ~.,data=myX,weights=rep(1,length(myYFactor)),size=10,
         decay=lambda,MaxNWts=10000,maxit=250)
NNPred2=predict(fit2,newdata=mymaintestX, type = "class")
resultNN <- cbind(test[1],prediction=as.numeric(NNPred2))
resultNN <- rbind(resultNN, data.frame(cbind(bidder_id = missingTest,prediction= 0)))
write.csv(resultNN, file = "submissionNN.csv", quote = FALSE, row.names = FALSE)


library(randomForest)
RFmodel2 = randomForest(x = myX, y = myYFactor, ntree = 1000)
RFPred2 = predict(RFmodel2, mymaintestX, type = "class")
RFProb2 = predict(RFmodel2, mymaintestX, type = "prob")[,2]
resultRF <- cbind(test[1],prediction=as.numeric(NNPred2))
resultRF <- rbind(resultRF, data.frame(cbind(bidder_id = missingTest,prediction= 0)))
write.csv(resultRF, file = "submissionRF.csv", quote = FALSE, row.names = FALSE)

adamodel2 = boosting(myYfactor~.,data=adaTrain,mfinal=50,coeflearn ="Freund", control=rpart.control(maxdepth=10))
adaPred2 = predict(adamodel2, newdata=mymaintestX) 
adaProb2 = adaPred2$prob[,2] 


LRmodel2 = glm(myYtrain~ ., family = binomial(logit),data = LRtrain)
LRmodelstep2 = step(LRmodel2,k=3,direction = "both")
LRProb2 = predict(LRmodelstep2, newdata = data.frame(mymaintestX), type = "response")
  
stackingpredict <- cbind(svm2.prob,NNPred2,RFProb2,adaProb2,LRProb2)
finalstackpredict <- weight1[1]*as.numeric(svm2.prob) + weight1[2]*as.numeric(NNPred2) + weight1[3]*as.numeric(RFProb2) + weight1[4]*as.numeric(adaProb2) +weight1[5]*as.numeric(LRProb2) 
resultstack <- cbind(test[1],prediction=as.numeric(finalstackpredict))
resultstack <- rbind(resultstack, data.frame(cbind(bidder_id = missingTest,prediction= 0)))
write.csv(resultstack, file = "submissionstack.csv", quote = FALSE, row.names = FALSE)
