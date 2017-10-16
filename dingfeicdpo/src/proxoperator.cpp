#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

//' Using the method of "Proximal Operator" to get the beta of Lasso problem
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
Rcpp::List proxoperator( arma::vec& y, arma::mat& X,  double& lambda,  const int& maxinteration ){
  double epsilon = 0.1 ;
  
  int p = X.n_cols;
  int N = X.n_rows;
  double oldbeta;
  int j ;
  int i = 0 ;
  vec beta(p);
  beta = zeros<vec>(p);
  vec objectvalue= zeros<vec>(maxinteration);
  
  vec vepsilon= arma::ones<vec>(N) * epsilon;
  vec eigenvalue = eig_sym(X.t()*X);
  vec ggradient ;
  double M = max(eigenvalue)/N;
  double z ;
  double H = lambda/M;
  vec tempbeta ;
  
  
  do {
    tempbeta = beta;
    ggradient = beta+(X.t()*(y-X*beta))/(N*M);
    for(j=0; j<p;j++){

      if (ggradient(j) > H)
        beta(j) = ggradient(j)-H;
      else if (ggradient(j) < -H)
        beta(j) = ggradient(j)+H;
      else 
        beta(j)=0;
    
      }
    objectvalue(i) = sum((y-X*beta)%(y-X*beta))/(2*N)+lambda*sum(abs(beta));
    i = i +1;
  } while (i < maxinteration);
  
  
  return Rcpp::List::create(Rcpp::Named("proxoperator_beta")=beta,
                            Rcpp::Named("proxoperator_objectvalue")=objectvalue);;
}




