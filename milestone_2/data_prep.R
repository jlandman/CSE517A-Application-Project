#load libraries and read in data
library(caret)
library(kernlab)
library(dplyr)
library(plot3D)
set.seed(1234)
datsite <- url("https://raw.githubusercontent.com/jlandman/CSE517A-Application-Project/master/dataset/OnlineNewsPopularity/OnlineNewsPopularity.csv")
data <- read.csv(datsite)
data <- data[complete.cases(data),]
rand <- sample.int(39644, size = 39644, replace = FALSE, prob = NULL)
data <- data[rand,]
data <- data[1:3000,] #Training for the whole dataset is computationally intensive and time consuming so we show results for a random partition of the dataset
data$logshares<- log(data$shares)
data <- select(data,-url,-shares)

#create train and test splits
train <- createDataPartition(data$logshares,p=0.8,list=FALSE)
xTr <- select(data,-logshares)[train,]
yTr <- data$logshares[train]
fulltrain <- data.frame(xTr,yTr)
train1 <- createDataPartition(yTr,p=0.5,list=FALSE)
xTr1 <- xTr[train1,]
yTr1 <- yTr[train1]
xTr2 <- xTr[-train1,]
yTr2 <- yTr[-train1]
xTe <- select(data,-logshares)[-train,]
yTe <- data$logshares[-train]
train1 <- data.frame(xTr1,yTr1)
train2 <- data.frame(xTr2,yTr2)
test <- data.frame(xTe,yTe)

splits1 <- createFolds(yTr1, k = 10, returnTrain = TRUE)
splits2 <- createFolds(yTr2, k = 10, returnTrain = TRUE)