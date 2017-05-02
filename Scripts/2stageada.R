setwd( "\\\\prism.nas.gatech.edu/hreddivari3/vlab/documents/cdaproject/facebook3")
rm(list = ls())
require(xgboost)
require(randomForest)
require(e1071)
require(TunePareto)
require(ROCR)
require(adabag)
load("extractedFeaturesinit3.Rda")
# kfold stratified
CV_split <- function(train_label, K){
  result = generateCVRuns(train_label, ntimes = 1, nfold = K, stratified = T)
  result[[1]]
}
set.seed(123)
evaluation = function(probs, target, eps = 1e-15){
  probs[probs < eps] = eps
  probs[probs > 1- eps] = eps
  pred = prediction(probs, target)
  perf = performance(pred, 'auc')
  attributes(perf)['y.values'][[1]][[1]]
}

#df = read.csv('train.csv',header = FALSE)
#names(df) <- c("bidder_id","payment_account","address","outcome")
#df$bider_id = NULL

df <- data.frame(train[,!(attributes(train)$names %in% c("bidder_id","crime_median_p","crime_mean_pw","crime_mean_t","crime_mean_tw"))])
outcome <- df[,1]
df <- cbind(df,outcome)
df <- df[,-1]
TOTAL_RUNS = 10 # total rounds of CV 
K = 10 # K-fold stratified CV
bagSize = 20 # bagging size  
df$outcome <- as.numeric(df$outcome)
## model params 
boos = TRUE # adaboost
mfinal = 15
coeflearn = 'Breiman'
ntree = 60 # rf
mtry = 9
gamma = 0.01 # svm
cost = 3
train <- df
train$merchandise <- as.numeric(train$merchandise)
##### CV process #####
total_scores = rep(0,TOTAL_RUNS)
for(r in 1:TOTAL_RUNS){
  #r = 1
  CV_index = CV_split(df$outcome, K)
  auc_scores = rep(0,K)
  cat("Scores: ")
  for (cv in 1:K){
    # cv = 1
    cat("CV: ", cv, "\n")
    valid_index = CV_index[[cv]]
    
    template = 1:nrow(df)
    valid_template = template %in% valid_index
    
    valid_x = train[valid_template,]  # validating data set
    train_x = train[!valid_template,]  # training data set
    
    # not scaled, for rf 
    train_x_rf = train_x
    valid_x_rf = valid_x
    
    # log transform
    #train_x[,1:(ncol(train_x)-1)] = log(1.0 + train_x[,1:(ncol(train_x)-1)])
    #valid_x[,1:(ncol(valid_x)-1)] = log(1.0 + valid_x[,1:(ncol(valid_x)-1)])
    
    trind = 1:nrow(train_x)
    #teind = 1:nrow(valid_df) no use 
    
    ### start bagging ###
    baggingRuns = 1:bagSize
    pred_final = 0
    for (z in baggingRuns) {
	#z = 1
      cat(z, ' ')
      #bag_index = sample(trind,size=as.integer(length(trind)),replace=T)
      #OOB_index = setdiff(trind,bag_index)
      
      ### stratified split bag and OOB
      stra_index = CV_split(train_x[,ncol(train_x)], 4) # 75%
      OOB_index_temp = stra_index[[1]]
      OOB_index = trind %in% OOB_index_temp
      bag_index = setdiff(trind,OOB_index)
      
      X_rf = train_x_rf[OOB_index,-ncol(train_x)]
      y_rf = train_x_rf[OOB_index, ncol(train_x)]
      
      X_svm = train_x[OOB_index,-ncol(train_x)]
      y_svm = train_x[OOB_index, ncol(train_x)]
      
      # train models on OOB sets
      rf_model = randomForest(x=X_rf, y=as.factor(y_rf), replace=T, ntree=ntree, do.trace=F, mtry=mtry)
	svm_model = svm(as.factor(y_svm)~., data = X_svm, gamma = gamma, cost = cost, class.weights=c('1'=0.1,'2'=1.0), 
                      probability = TRUE)
      
      # assign bagging sets to ADA
      X_ada = train_x[bag_index,-ncol(train_x)]
      y_ada = train_x[bag_index, ncol(train_x)]
      
      # predict rf and svm on bagging sets
      rf_pred = predict(rf_model, X_ada, type="prob")
      rf_pred = rf_pred[,2]
      svm_pred = predict(svm_model, X_ada, probability=TRUE)
      svm_pred = attr(svm_pred,'probabilities')[,2]
      
      # predict on valid sets 
      rf_pred_valid = predict(rf_model, valid_x_rf[,-ncol(valid_x_rf)], type="prob")
      rf_pred_valid = rf_pred_valid[,2]
      svm_pred_valid = predict(svm_model, valid_x[,-ncol(valid_x)], probability=TRUE)
      svm_pred_valid = attr(svm_pred_valid,'probabilities')[,2]
      
      ## train xgboost model with probabilities from previous preds 
      # combine new training and validating dfs
      train_ada = cbind(X_ada, temp=(rf_pred*svm_pred)^0.5, outcome =y_ada)
      
      valid_ada_X = cbind(valid_x[,-ncol(valid_x)], temp=(rf_pred_valid*svm_pred_valid)^0.5, outcome=valid_x[,ncol(valid_x)])
      
      # train ada
	outcome <- as.factor(train_ada$outcome)
 	train_ada <- train_ada[,-ncol(train_ada)]
	train_ada <- cbind(outcome,train_ada)
      ada_model = boosting(outcome~., data = train_ada, boos = boos, mfinal = 50, coeflearn = coeflearn, control = (minsplit = 0))
      
      # predict 
      pred = predict(ada_model, valid_ada_X, probability = TRUE)
      pred = pred$prob[,2]
      
      pred_final = pred_final + pred
      #pred_final = pred_final + rank(pred, ties.method = "random")
      
    } # end of bagging 
    #pred_final = seq(from=0, to=1, length.out=nrow(valid_x))[rank(pred_final, ties.method = "random")]
    pred_final = pred_final / z
    auc_scores[cv] = evaluation(pred_final, valid_x[,ncol(valid_x)])
    cat(auc_scores[cv], " ")
  } # end of CV
  cat("\n")
  print(auc_scores)
  cat("mean: ", mean(auc_scores), "sd: ", sd(auc_scores))
  total_scores[r] = mean(auc_scores)
}

print("\nTOTAL RUns SCORES: \n")
print(total_scores)
print(mean(total_scores))
save.image("2stageada.RData")
##### SUBMISION! #####
## data loading and preparation
df = read.csv('train.csv')
df$bider_id = NULL
submit_df = read.csv('test.csv')
test_id = as.character(submit_df$bider_id)
submit_df$bider_id = NULL
x = as.matrix(df)
x = matrix(as.numeric(x),nrow(x),ncol(x))
submit_x = as.matrix(submit_df)
submit_x = matrix(as.numeric(submit_x),nrow(submit_x),ncol(submit_x))

x_rf = x # rf do not scale
x[,1:(ncol(x)-1)] = (1.0 + x[,1:(ncol(x)-1)])
submit_x_rf = submit_x
# dummy col 
submit_x$outcome = 1
submit_x$outcome[1] = 0
submit_x$outcome = as.factor(submit_x$outcome)


## model params 
ntree = 60 # rf
mtry = 9
gamma = 0.005 # svm
cost = 2
boos = TRUE # ada
mfinal = 15
coeflearn = 'Breiman'
x_rf <- train[,-1]
### start bagging 
trind = 1:nrow(x)

baggingSize = 1:80
pred_final = 0
for (z in baggingSize) {
  cat(z, " \n")
  
  stra_index = CV_split(x[,ncol(x)], 4) # 75%
  OOB_index_temp = stra_index[[1]]
  OOB_index = trind %in% OOB_index_temp
  bag_index = setdiff(trind,OOB_index)
  
  X_rf = x_rf[OOB_index,-ncol(x_rf)]
  y_rf = x_rf[OOB_index, ncol(x_rf)]
  
  X_svm = x[OOB_index,-ncol(x)]
  y_svm = x[OOB_index, ncol(x)]
  
  # train models on OOB sets
  rf_model = randomForest(x=X_rf[which(complete.cases(X_rf)==TRUE),], y=as.factor(y_rf[which(complete.cases(X_rf)==TRUE)]), replace=T, ntree=ntree, do.trace=F, mtry=mtry, na.action =na.fail)
  svm_model = svm(as.factor(y_svm)~., data = X_svm, gamma = gamma, cost = cost, class.weights=c('1'=0.1,'2'=1.0),
                  probability = TRUE)
  
  # assign bagging sets to XGB
  X_ada = x[bag_index,-ncol(x)]
  y_ada = x[bag_index, ncol(x)]
  
  # predict rf and svm on bagging sets
  rf_pred = predict(rf_model, X_ada, type="prob")
  rf_pred = rf_pred[,2]
  svm_pred = predict(svm_model, X_ada, probability=TRUE)
  svm_pred = attr(svm_pred,'probabilities')[,2]
  
  # predict on submit sets 
  rf_pred_submit = predict(rf_model, submit_x_rf, type="prob")
  rf_pred_submit = rf_pred_submit[,2]
  svm_pred_submit = predict(svm_model, submit_x[,-ncol(submit_x)], probability=TRUE)
  svm_pred_submit = attr(svm_pred_submit,'probabilities')[,2]
  
  ## train ada model with probabilities from previous preds 
  train_ada = cbind(X_ada, temp=(rf_pred*svm_pred)^0.5, outcome =y_ada)
  
  submit_ada_X = cbind(submit_x[,-ncol(submit_x)], temp=(rf_pred_submit*svm_pred_submit)^0.5, outcome=submit_x[,ncol(submit_x)])
  
  # train ada and predict on submit_ada_x
  ada_model = boosting(outcome~., data = train_ada, boos = boos, mfinal = 15, coeflearn = coeflearn)
  pred = predict(ada_model, submit_ada_X, probability = TRUE)#, probability = TRUE)
  pred = pred$prob[,2]
  
  pred_final = pred_final + pred
  
} # end of bagging 
pred_final = pred_final / z

### prepare submission 
submitFile = read.csv('sampleSubmission.csv')
submitFile$bidder_id = as.character(submitFile$bidder_id)
for (i in 1:4630){
  submitFile$prediction[submitFile$bidder_id == test_id[i]] = pred_final[i]
}
write.csv(submitFile, 'bag_adaboost_pred.csv', row.names = FALSE)  

tenrunAUC <- c(0.8185185, 0.9666344, 0.9284333, 0.8781431, 0.9473404, 0.8856383, 0.9066489,
0.9622340, 0.9319149,0.9193122, 0.8853965, 0.8423598, 0.9719536, 0.9398936, 0.8752660, 0.8311170, 0.9500000, 0.9574468, 0.8872340,
0.9333333, 0.8636364, 0.9182785, 0.9647002, 0.9164894, 0.9303191, 0.8553191, 0.8731383, 0.9367021, 0.9000000,0.8164021, 0.8612186,
0.9342360, 0.9429400, 0.9510638, 0.8393617, 0.8877660, 0.8664894, 0.9925532, 0.9553191,0.8259259, 0.8926499, 0.9105416, 0.8322050,
0.9297872, 0.9632979, 0.8920213, 0.9404255, 0.9558511, 0.9390957, 0.8941799, 0.9552708, 0.8409091, 0.8996615, 0.9420213, 0.9053191,
0.9494681, 0.9069149, 0.9531915, 0.8930851, 0.9444444, 0.8974855, 0.9564797, 0.9647002, 0.8446809, 0.9013298, 0.9252660, 0.9521277,
0.8212766, 0.8648936, 0.9275132, 0.8554159, 0.9385880, 0.9052224, 0.9664894, 0.9324468, 0.8404255, 0.9457447, 0.9345745, 0.8414894,
0.8793651, 0.8457447, 0.9453578, 0.9361702, 0.9898936, 0.9015957, 0.8119681, 0.9239362, 0.9404255, 0.9468085,0.9333333, 0.9078820,
0.8438104, 0.8931335, 0.9138298, 0.9425532, 0.9000000, 0.8696809, 0.9143617, 0.9393617)


