
#ifdef USE_FLOAT
typedef float RealNum;
#else
#ifdef USE_DOUBLE
typedef double RealNum;
#else
#ifdef FLYCHECK
typedef double RealNum;
#else
COMPILATION ERROR: You need to either provide the pre-processor direvtive USE_FLOAT or USE_DOUBLE! See the package.yaml file of grenade!;
typedef double RealNum;                /* To prevent other errors! */
#endif
#endif
#endif
