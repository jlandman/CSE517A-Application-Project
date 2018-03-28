source("data_prep.R")
source("nlpd.R")
source("polynomial.R")
source("rbf.R")
source("best.R")

#Test whether the three are significantly different
lp <- t.test(cverrs$linear,cverrs$polynomial,paired=TRUE)
lp
lr <- t.test(cverrs$linear,cverrs$rbf,paired=TRUE)
lr
pr <- t.test(cverrs$polynomial,cverrs$rbf,paired=TRUE)
pr

#Train on all data and calculate test error for chosen model
model <- gausspr(yTr~.,data=fulltrain,kernel=k,variance.model=TRUE,kpar=par)
nlpdtrain <- nlpd(model,xTr,yTr)
nlpdtrain
nlpdtest <- nlpd(model,xTe,yTe)
nlpdtest
maetrain <- mean(abs(yTr-predict(model,xTr)))
maetrain
maetest <- mean(abs(yTe-predict(model,xTe)))
maetest
