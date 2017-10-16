#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

//' Using the method of "coordinate descent" to get the beta of Lasso problem
//' @param y the response data
//' @param X the design matrix
//' @param lambda the coefficient of the panalty
//' @param maxinteration the maximum of interation
//' @return the list of the beta and object function value in each interation
//' @examples 
//' the data must be regularization before using the function in the package
//' require(dingfeicdpo)
//' standvector = function(y){
//'return(y-mean(y))
//'}
//'standmatrix = function(X){
//'  n=dim(X)[1]
//'  p=dim(X)[2]
//'  meanvec=c()
//'  squarevec=c()
//'  for(i in 1:p){
//'    X[,i]=X[,i]-mean(X[,i])
//'  }
//'  for(j in 1:p){
//'    squarevec=append(squarevec,sum(X[,j]*X[,j]))
//'  }
//'  for(k in 1:p){
//'    X[,k]=X[,k]/(sqrt(squarevec[k]/n))
//'  }
//'  return(X)
//'}
//'n = 100
//'p = 500
//'sigma_noise = 0.5
//'beta = rep(0, p)
//'beta[1:6] = c(5,10,3,80,90,10)
//'X = matrix(rnorm(n*p,sd=10), nrow = n, ncol=p)
//'X_stand=standmatrix(X)
//'y = X_stand %*% beta + rnorm(n, sd = sigma_noise)
//'y_stand=standvector(y)
//'lambda = 0.2
//'maxinteration = 500
//'# Compute Coordinate descent 
// betaHat1 = cdescent(y_stand, X_stand, lambda,maxinteration)
//'# Compute Proximal Operator
//'    betaHat2 = proxoperator(y_stand, X_stand, lambda,maxinteration)
// [[Rcpp::export]]
Rcpp::List cdescent(const arma::vec& y, const arma::mat& X, const double& lambda,const int& maxinteration){
  double epsilon = 0.1 ;
  
  int p = X.n_cols;
  int N = X.n_rows;
  double oldbeta;
  int j ;
  int i = 0;
  vec objectvalue= zeros<vec>(maxinteration);
  vec beta(p);
  beta = zeros<vec>(p);
  
  vec vepsilon= ones<vec>(N) * epsilon;
  vec r = y - X * beta;
  do {
    for(j=0; j<p;j++){
      oldbeta = beta(j);
      
      if (sum(r%X.col(j))/N+beta(j) > lambda)
        beta(j) = sum(r%X.col(j))/N+beta(j) - lambda;
      else if (sum(r%X.col(j))/N+beta(j) < -lambda)
        beta(j) = sum(r%X.col(j))/N+beta(j) +lambda;
      else 
        beta(j)=0;
    
      
      r= r+X.col(j)*(oldbeta-beta(j));
      }
    objectvalue(i) = sum(r%r)/(2*N)+lambda*sum(abs(beta));
    i = i + 1;
  } while (i < maxinteration);
  
  return Rcpp::List::create(Rcpp::Named("cdescent_beta")=beta,
                            Rcpp::Named("cdescent_objectvalue")=objectvalue);
}
  
  
  
  
