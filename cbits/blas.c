#include "type.h" 

void dgemm_(char *, char *, int *, int *, int *,
           double *, const double *, int *, const double *,
           int *, double *, double *, int *);

void dgemm_direct(int ta, int tb, int m, int n, int k,
  double alph, RealNum* ap, int aXc, RealNum* bp, int bXc, double bet,
  RealNum* rp, int rXc) {
    int lda = aXc;
    int ldb = bXc;
    int ldc = rXc;
    double alpha = alph;
    double beta = bet;
    dgemm_(ta?"T":"N",tb?"T":"N",&m,&n,&k,&alpha,ap,&lda,bp,&ldb,&beta,rp,&ldc);
}

void dgemv_(char *, int *, int *, 
  double *, const double *, int *,
  const double *, int *,
  double *, double *, int *);

void dgemv_direct(int ta, int m, int n,
  double alph, RealNum* ap, int aXc,
  RealNum* xp, int incX,
  double bet, RealNum* yp, int incY) {
    int lda = aXc;
    int iX = incX;
    int iY = incY;
    double alpha = alph;
    double beta = bet;
    dgemv_(ta?"T":"N",&m,&n,&alpha,ap,&lda,xp,&iX,&beta,yp,&iY);
}


void dger_(int *, int *, 
  double *, const double *, int *,
  const double *, int *,
  double *, int *);

void dger_direct(int m, int n,
  double alph, RealNum* xp, int incX,
  RealNum* yp, int incY,
  RealNum* ap, int aXc) {
    int lda = aXc;
    int iX = incX;
    int iY = incY;
    double alpha = alph;
    dger_(&m,&n,&alpha,xp,&iX,yp,&iY,ap,&lda);
}


