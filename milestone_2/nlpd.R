#Function to compute the negative log predictive density

nlpd <- function(model,xTe,yTe){
  n <- length(yTe)
  sigma <- predict(model,xTe,type="variance")
  mu <- predict(model,xTe)
  if(sum(sigma>0)<n){
    NA
  }
  else{
    -sum(log(2*pi*sigma)+(((yTe-mu)^2)/(2*sigma)))/n
  }
}