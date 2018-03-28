degrees <- 1:4
scales <- seq(0,4,1)
offsets <- -2:2
smallestpoly <- 10000
bestdeg <- NA
bestscal <- NA
bestoff <- NA
polyerrs <- numeric()
len <- length(degrees)*length(scales)*length(offsets)
polyplot <- data.frame(degrees=rep(NA,len),scales=rep(NA,len), offsets=rep(NA,len), nlpd=rep(NA,len))
i <- 1
for(deg in degrees){
  for(scal in scales){
    for(off in offsets){
      for(s in 1:length(splits1)){
        val <- splits1[[s]]
        polyGP <- gausspr(yTr1~.,data=train1[-val,],kernel="polydot",kpar=list(degree=deg,scale=scal,offset=off),variance.model=TRUE)
        polyerrs[s] <- nlpd(polyGP,xTr1[val,],yTr1[val])
      }
      err <- mean(polyerrs,na.rm=TRUE)
      polyplot$degrees[i] <- deg
      polyplot$scales[i] <- scal
      polyplot$offsets[i] <- off
      if(sum(is.na(polyerrs))>0){
        #this combination wont work so do nothing
        polyplot$nlpd[i] <- NA
      }
      else {
        if(err<smallestpoly){
          bestdeg <- deg
          bestscal <- scal
          bestoff <- off
          smallestpoly <- err
        }
        polyplot$nlpd[i] <- err
      }
      i <- i + 1
    }
  }
}
polyplot <- polyplot[complete.cases(polyplot),]
scatter3D(polyplot$degree, polyplot$scale, polyplot$offset, colvar=polyplot$nlpd, clab = c("NLPD"),xlab="Degree",ylab="Scale",zlab="Offset")
points3D(bestdeg,bestscal,bestoff, add=TRUE, pch=16)
points3D(bestdeg,bestscal,bestoff, add=TRUE, pch=8)
text3D(bestdeg,bestscal,bestoff+.2, add=TRUE, label=paste0("(",bestdeg,",",bestscal,",",bestoff,")"))
