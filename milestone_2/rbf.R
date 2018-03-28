sigmas <- seq(0,1,0.1)
bestsig <- NA
smallestrbf <- 10000
rbferrs <- numeric()
sigmaplot <- data.frame(sigmas=sigmas,nlpd=rep(NA,length(sigmas)))
i <- 1
for(sig in sigmas){
  for(s in 1:length(splits1)){
    val <- splits1[[s]]
    rbfGP <- gausspr(yTr1~.,data=train1[-val,],kernel="rbfdot",kpar=list(sigma=sig),variance.model=TRUE)
    rbferrs[s] <- nlpd(rbfGP,xTr1[val,],yTr1[val])
  }
  err <- mean(rbferrs,na.rm=TRUE)
  if(err < smallestrbf){
    bestsig <- sig
    smallestrbf <- mean(rbferrs)
  }
  sigmaplot$nlpd[i] <- err
  i <- i + 1
}
sigmaplot <- sigmaplot[complete.cases(sigmaplot),]
plot(sigmaplot$sigma, sigmaplot$nlpd, xlab="Sigma", ylab="NLPD")
