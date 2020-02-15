
rm(list=ls())
library(pROC)
library(tictoc)
load_mnist <- function() {
  load_image_file <- function(filename) {
    ret = list()
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    ret$n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
    ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    ret
  }
  load_label_file <- function(filename) {
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    y = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    y
  }
  train <<- load_image_file("train-images-idx3-ubyte")
  test <<- load_image_file('t10k-images-idx3-ubyte')
  
  train$y <<- load_label_file('train-labels-idx1-ubyte')
  test$y <<- load_label_file('t10k-labels-idx1-ubyte')  
}


show_digit <- function(arr784, col=gray(128:1/128), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}
load_mnist()
###############################
# reduce to digit 1 against digit 2
###############################
digit1=2
digit2=9
n=250
ntest = 1000
ind.train = which(train$y==digit1 | train$y==digit2)[1:n]
ind.test= which(test$y==digit1 | test$y==digit2)[1:ntest]
train$x=train$x[ind.train,]
train$y=train$y[ind.train]
test$x=test$x[ind.test,]
test$y=test$y[ind.test]

train$y[train$y==digit1] = train$y[train$y==digit1]*0 + 1
train$y[train$y==digit2] = train$y[train$y==digit2]*0 - 1

test$y[test$y==digit1] = test$y[test$y==digit1]*0 + 1
test$y[test$y==digit2] = test$y[test$y==digit2]*0 - 1

tic()
##################################
# add a column of 1
##################################
train$x=cbind(train$x,as.vector(rep(1,times=dim(train$x)[1])))
test$x=cbind(test$x,as.vector(rep(1,times=dim(test$x)[1])))
##################################
# center
##################################
test$x = scale(test$x,center=as.vector(colMeans(train$x)),scale=F)
train$x = scale(train$x,scale=F)

##################################
# Kernel
##################################
d = 4
ker = function(x,y)
  return((1+sum(x%*%y))^d)

lambda=0.01
kk=outer(1:n,1:n,Vectorize(function(i,j) ker(train$x[i,],train$x[j,])))
alpha = solve(kk + lambda*n*diag(rep(1,n)))%*%(train$y)
toc()
tic()
test_out = rep(0,dim(test$x)[1])
for(i in 1:dim(test$x)[1]){
  for(j in 1:n){
    test_out[i] <- test_out[i] + alpha[j]*ker(test$x[i,],train$x[j,])
  }
}

accuracy = sum(sign(test_out) == test$y)/length(test$y)
toc()
paste0("N train = ", n, "N test = ", ntest, " accuracy = ", accuracy, " number correct = ", sum(sign(test_out) == test$y))