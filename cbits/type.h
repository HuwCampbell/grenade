
#ifdef USE_FLOAT
typedef float F;
#else
#ifdef USE_DOUBLE
typedef double F;
#else
#ifdef FLYCHECK
typedef double F;
#else
COMPILATION ERROR: You need to either provide the pre-processor direvtive USE_FLOAT or USE_DOUBLE! See the package.yaml file of grenade!;
typedef double F;                /* To prevent other errors! */
#endif
#endif
#endif
