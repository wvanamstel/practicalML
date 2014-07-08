library(caret)
library(doMC)
registerDoMC(2)  #allow for multicore calcs

training <- read.csv("../data/pml-training.csv")
testing <- read.csv("../data/pml-testing.csv")

##remove empty and NA columns from data set
no_na <- sapply(training, function(x) all(!is.na(x)))
train_pp <-training[no_na]
not_empty <- sapply(train_pp, function(x) all(x != ""))
train_pp <- train_pp[not_empty]
#remove user_name, cvtd_timestamp and new_window columns, i.e. non sensor data
#and column magnet_belt_z which does not exist in test set
train_pp <- train_pp[,-c(1:6, 20)]
 

### principal component analysis
pc_a <- prcomp(train_pp[-53], center=TRUE, scale=TRUE)
PC_var <- pc_a$sd^2/sum(pc_a$sd^2)
plot(PC_var)

### first two components seem significant, plot clusters:
require(ggplot2)
ggplot(data=NULL, aes(x=pc_a$x[,1], y=pc_a$x[,2], color=train_pp$classe)) +
  geom_point(size=3,alpha=0.55)+ guides(colour = guide_legend(override.aes = list(alpha = 1)))

##create a training and validation set
set.seed(23)
in_train <- createDataPartition(y=train_pp$classe, p=0.70, list=FALSE)
validation <- train_pp[-in_train,]
train_pp <- train_pp[in_train,]

#some columns are not relevant for modelling, find correlated predictors
M <- abs(cor(train_pp[,-53]))
diag(M) <- 0
which(M > 0.80, arr.ind=TRUE) ##there are many correlated predictors
require(corrgram)
corrgram(M[1:15,1:15], main="Correlogram of first 15 predictors", upper.panel=NULL)

######################################################################
## initial model runs w/o preprocessing data
######################################################################
#construct a trainControl object to allow for parallel processing
require(doMC)
registerDoMC(2)  #allow for multicore calcs
tr_control <- trainControl(allowParallel = TRUE)
##regression tree model
model_rpart <- train(train_pp$classe ~ ., trControl= tr_control, method="rpart", data=train_pp)
##support vector machine model
model_svm <- svm(train_pp$classe ~ ., data=train_pp)
##random forest with 4 fold resampling
tr_control <- trainControl(method="cv", number=4, allowParallel=TRUE)
model_rf <- train(train_pp$classe ~ ., trControl=tr_control, method="rf", data=train_pp)

#now use caret package to preprocess the data
train_ind <- sample(nrow(train_pp), 4000)
train_small <- train_pp[train_ind,]
pre_proc <- preProcess(train_small[,-54], method="pca", thresh=0.95)
train_pre_proc <- predict(pre_proc, train_small[,-54])
#random forest with preprocessad data and sampled training data set
mod_rf_pp <- train(train_small$classe ~ ., trControl= tr_control, method="rf", data=train_pre_proc)
#much less accuracy (0.89), using 4000 samples and 26 predictors

######################################################################
## use models on validation set
######################################################################
pred_rf <- predict(model_rf, validation)
sum(pred_rf == validation$classe)/length(pred_rf)
pred_svm <- predict(model_svm, validation)
sum(pred_svm == validation$classe)/length(pred_svm)
pred_rpart <- predict(model_rpart, validation)
sum(pred_rpart == validation$classe)/length(pred_rpart)

validation$pred_right <- pred_rf == validation$classe
confusionMatrix(pred_rf, validation$classe)

######################################################################
## use models on test set
######################################################################
#make sure that the test set contains the same columns as the training set
no_na <- sapply(testing, function(x) all(!is.na(x)))
test_pp <-testing[no_na]
not_empty <- sapply(test_pp, function(x) all(x != ""))
test_pp <- test_pp[not_empty]
#remove user_name, cvtd_timestamp and new_window columns, i.e. non sensor data
test_pp <- test_pp[,-c(1:6,20)]

test_rpart <- predict(model_rpart,test_pp)
test_svm <- predict(model_svm, test_pp)
test_rf <- predict(model_rf, test_pp)

######################################################################
## write result files
###################################################################### 
test_char <- as.character(test_rf)
setwd("../data/results/")   
pml_write_files = function(x){
   n = length(x)
   for(i in 1:n){
       filename = paste0("problem_id_",i,".txt")
       write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
   }
}

pml_write_files(test_char)
