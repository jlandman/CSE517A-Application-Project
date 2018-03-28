#Pick best model
cverrs <- data.frame(linear=rep(NA,length(splits2)),polynomial=rep(NA,length(splits2)),rbf=rep(NA,length(splits2)))
for(s in 1:length(splits2)){
  val <- splits2[[s]]
  linGP <- gausspr(yTr2~.,data=train2[-val,],kernel="vanilladot",variance.model=TRUE)
  cverrs$linear[s] <- nlpd(linGP,xTr2[val,],yTr2[val])
  polyGPBest <- gausspr(yTr2~.,data=train2[-val,],kernel="polydot",variance.model=TRUE,kpar=list(degree=bestdeg,scale=bestscal,offset=bestoff))
  cverrs$polynomial[s] <- nlpd(polyGP,xTr2[val,],yTr2[val])
  polyGPBest <- gausspr(yTr2~.,data=train2[-val,],kernel="rbfdot",variance.model=TRUE,kpar=list(sigma=bestsig))
  cverrs$rbf[s] <- nlpd(rbfGP,xTr2[val,],yTr2[val])
}
cverr <- colMeans(cverrs,na.rm=TRUE)

inds = which(cverr == min(cverr), arr.ind=TRUE)
cnames = names(inds)[1]

if(cnames=="linear"){
  k="vanilladot"
  par="automatic"
}
if(cnames=="polynomial"){
  k="polydot"
  par=list(degree=bestdeg,scale=bestscal,offset=bestoff)
}
if(cnames=="rbf"){
  k="rbfdot"
  par=list(sigma=bestsig)
}

