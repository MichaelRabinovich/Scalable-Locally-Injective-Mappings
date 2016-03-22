#ifndef __PardisoSolver__
#define __PardisoSolver__

#include <vector>
#include <Eigen/Core>

 typedef int IndexType;
 typedef double ScalarType;


 extern "C" {
 /* PARDISO prototype. */
 void pardisoinit (void   *, int    *,   int *, int *, double *, int *);
 void pardiso     (void   *, int    *,   int *, int *,    int *, int *,
                   double *, int    *,    int *, int *,   int *, int *,
                   int *, double *, double *, int *, double *);
 void pardiso_chkmatrix  (int *, int *, double *, int *, int *, int *);
 void pardiso_chkvec     (int *, int *, double *, int *);
 void pardiso_printstats (int *, int *, double *, int *, int *, int *,
                          double *, int *);
 }
 //#include "types.h"
 //template <typename IndexType, typename ScalarType>
 class PardisoSolver
 {
 public:
   PardisoSolver(Eigen::VectorXi& ia, Eigen::VectorXi& ja, Eigen::VectorXd& a) ;
   ~PardisoSolver();
   void set_type(int _mtype);
   void init();
   void set_pattern() {numRows = ia.rows()-1;};
   void set_pattern(const std::vector<IndexType> &II,
                    const std::vector<IndexType> &JJ,
                    const std::vector<ScalarType> SS);
  void set_pattern(Eigen::VectorXi& ai,
                    Eigen::VectorXi& aj,
                    Eigen::VectorXd& vals) {
      ia = ai; ja = aj; a = vals; numRows = ia.rows()-1;
  };
   void analyze_pattern();
   bool factorize();
   void solve(Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> &rhs,
              Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> &result);
   void free_numerical_factorization_memory();
   void update_a(const std::vector<ScalarType> SS);

   Eigen::VectorXi& ia; Eigen::VectorXi& ja;
   Eigen::VectorXd& a;
   std::vector<Eigen::VectorXi> iis;
 protected:

   int numRows;

   //pardiso stuff
   int mtype;       /* Real symmetric matrix */// olga: was -2. Is Symmetric positive definite.
   int nrhs = 1;     /* Number of right hand sides. */
   /* Internal solver memory pointer pt, */
   /* 32-bit: int pt[64]; 64-bit: long int pt[64] */
   /* or void *pt[64] should be OK on both architectures */
   void *pt[64];
   /* Pardiso control parameters. */
   int iparm[64];
   double   dparm[64];
   int maxfct, mnum, phase, error, msglvl, solver =0;
   /* Number of processors. */
   int      num_procs;
   /* Auxiliary variables. */
   char    *var;
   int i, k;
   double ddum;          /* Double dummy */
   int idum;         /* Integer dummy. */

   int numUniqueElements;

};

//#include "PardisoSolver.hpp"

#endif /* defined(__CurlFree__PardisoSolver__) */
