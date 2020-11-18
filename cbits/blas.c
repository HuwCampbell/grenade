#include "type.h"

void dgemm_(char *, char *, int *, int *, int *,
           double *, const double *, int *, const double *,
           int *, double *, double *, int *);

void dgemm_direct(int ta, int tb, int m, int n, int k,
  double alph, double* ap, int aXc, double* bp, int bXc, double bet,
  double* rp, int rXc) {
    int lda = aXc;
    int ldb = bXc;
    int ldc = rXc;
    double alpha = alph;
    double beta = bet;
    dgemm_(ta?"T":"N",tb?"T":"N",&m,&n,&k,&alpha,ap,&lda,bp,&ldb,&beta,rp,&ldc);
}

void sgemm_(char *, char *, int *, int *, int *,
           float *, const float *, int *, const float *,
           int *, float *, float *, int *);

void sgemm_direct(int ta, int tb, int m, int n, int k,
  float alph, float* ap, int aXc, float* bp, int bXc, float bet,
  float* rp, int rXc) {
    int lda = aXc;
    int ldb = bXc;
    int ldc = rXc;
    float alpha = alph;
    float beta = bet;
    sgemm_(ta?"T":"N",tb?"T":"N",&m,&n,&k,&alpha,ap,&lda,bp,&ldb,&beta,rp,&ldc);
}


void dgemv_(char *, int *, int *,
  double *, const double *, int *,
  const double *, int *,
  double *, double *, int *);

void dgemv_direct(int ta, int m, int n,
  double alph, double* ap, int aXc,
  double* xp, int incX,
  double bet, double* yp, int incY) {
    int lda = aXc;
    int iX = incX;
    int iY = incY;
    double alpha = alph;
    double beta = bet;
    dgemv_(ta?"T":"N",&m,&n,&alpha,ap,&lda,xp,&iX,&beta,yp,&iY);
}

void sgemv_(char *, int *, int *,
  float *, const float *, int *,
  const float *, int *,
  float *, float *, int *);

void sgemv_direct(int ta, int m, int n,
  float alph, float* ap, int aXc,
  float* xp, int incX,
  float bet, float* yp, int incY) {
    int lda = aXc;
    int iX = incX;
    int iY = incY;
    float alpha = alph;
    float beta = bet;
    sgemv_(ta?"T":"N",&m,&n,&alpha,ap,&lda,xp,&iX,&beta,yp,&iY);
}


void dger_(int *, int *,
  double *, const double *, int *,
  const double *, int *,
  double *, int *);

void dger_direct(int m, int n,
  double alph, double* xp, int incX,
  double* yp, int incY,
  double* ap, int aXc) {
    int lda = aXc;
    int iX = incX;
    int iY = incY;
    double alpha = alph;
    dger_(&m,&n,&alpha,xp,&iX,yp,&iY,ap,&lda);
}


void sger_(int *, int *,
  float *, const float *, int *,
  const float *, int *,
  float *, int *);

void sger_direct(int m, int n,
  float alph, float* xp, int incX,
  float* yp, int incY,
  float* ap, int aXc) {
    int lda = aXc;
    int iX = incX;
    int iY = incY;
    float alpha = alph;
    sger_(&m,&n,&alpha,xp,&iX,yp,&iY,ap,&lda);
}
