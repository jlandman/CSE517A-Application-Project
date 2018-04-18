#load libraries and read in data
library(dplyr)
library(neuralnet)
library(caret)
set.seed(1234)
datsite <- url("https://raw.githubusercontent.com/jlandman/CSE517A-Application-Project/master/dataset/OnlineNewsPopularity/OnlineNewsPopularity.csv")
data <- read.csv(datsite)
data <- data[complete.cases(data),]
rand <- sample.int(39644, size = 39644, replace = FALSE, prob = NULL)
data <- data[rand,]
data <- data[1:500,]
data$logshares<- log(data$shares)
data <- select(data,-url,-shares)
scaled <- as.data.frame(scale(data))#Z-Scale data
center <- unname(colMeans(data,na.rm=TRUE)[60])
scale <- unname(apply(data,2,sd,na.rm=TRUE)[60])
splits <- createFolds(scaled[,60], k = 10, returnTrain = TRUE)

#Cross Validation to Choose Number of Hidden Nodes
cv.error <- matrix(nrow=10,ncol=3)
#system.time(
for(i in 1:10){
  val <- splits[[i]]
  train.cv <- scaled[val,]
  test.cv <- scaled[-val,]
  
  nn1 <- neuralnet(f,data=train.cv,hidden=c(20,10),linear.output=T)
  preds <- compute(nn1,select(test.cv, -logshares))
  preds_rescaled <- (preds$net.result)*scale+center
  test.cv_rescaled <- (test.cv$logshares)*scale+center
  cv.error[i,1] <- sum((test.cv_rescaled - preds_rescaled)^2)/nrow(test.cv)
  
  nn2 <- neuralnet(f,data=train.cv,hidden=c(25,5),linear.output=T)
  preds <- compute(nn2,select(test.cv, -logshares))
  preds_rescaled <- (preds$net.result)*scale+center
  test.cv_rescaled <- (test.cv$logshares)*scale+center
  cv.error[i,2] <- sum((test.cv_rescaled - preds_rescaled)^2)/nrow(test.cv)
  
  nn3 <- neuralnet(f,data=train.cv,hidden=c(15,10),linear.output=T)
  preds <- compute(nn3,select(test.cv, -logshares))
  preds_rescaled <- (preds$net.result)*scale+center
  test.cv_rescaled <- (test.cv$logshares)*scale+center
  cv.error[i,3] <- sum((test.cv_rescaled - preds_rescaled)^2)/nrow(test.cv)
}
#)

colMeans(cv.error)

#Split into train and test sets
train <- scaled[1:449,]
test <- scaled[450:500,]

#Train neural network
n <- names(train)
form <- as.formula(paste("logshares ~", paste(n[!n %in% "logshares"], collapse = " + ")))
#system.time(
nn <- neuralnet(form,data=train,hidden=c(25,5),linear.output=T)
#)
plot(nn)

#Test neural network
preds <- compute(nn,select(test,-logshares))
preds_rescaled <- (preds$net.result)*scale+center
test_rescaled <- (test$logshares)*scale+center
MSE <- sum((test_rescaled - preds_rescaled)^2)/nrow(test)
MSE



