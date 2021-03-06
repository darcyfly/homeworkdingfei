% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{proxoperator}
\alias{proxoperator}
\title{Using the method of "Proximal Operator" to get the beta of Lasso problem}
\usage{
proxoperator(y, X, lambda, maxinteration)
}
\arguments{
\item{y}{the response data}

\item{X}{the design matrix}

\item{lambda}{the coefficient of the panalty}

\item{maxinteration}{the maximum of interation}
}
\value{
the list of the beta and object function value in each interation
}
\description{
Using the method of "Proximal Operator" to get the beta of Lasso problem
}
\examples{
the data must be regularization before using the function in the package
require(dingfeicdpo)
standvector = function(y){
return(y-mean(y))
}
standmatrix = function(X){
 n=dim(X)[1]
 p=dim(X)[2]
 meanvec=c()
 squarevec=c()
 for(i in 1:p){
   X[,i]=X[,i]-mean(X[,i])
 }
 for(j in 1:p){
   squarevec=append(squarevec,sum(X[,j]*X[,j]))
 }
 for(k in 1:p){
   X[,k]=X[,k]/(sqrt(squarevec[k]/n))
 }
 return(X)
}
n = 100
p = 500
sigma_noise = 0.5
beta = rep(0, p)
beta[1:6] = c(5,10,3,80,90,10)
X = matrix(rnorm(n*p,sd=10), nrow = n, ncol=p)
X_stand=standmatrix(X)
y = X_stand \%*\% beta + rnorm(n, sd = sigma_noise)
y_stand=standvector(y)
lambda = 0.2
maxinteration = 500
# Compute Coordinate descent 
# Compute Proximal Operator
   betaHat2 = proxoperator(y_stand, X_stand, lambda,maxinteration)
}
