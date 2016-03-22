#include "PardisoSolver.h"
#include <igl/sortrows.h>
#include <igl/unique.h>


 using namespace std;
 //#define PLOTS_PARDISO


 //#define PLOTS_PARDISO
 //template <typename IndexType, typename ScalarType>
 //PardisoSolver<IndexType, ScalarType>::PardisoSolver()
 PardisoSolver::PardisoSolver(Eigen::VectorXi& ia, Eigen::VectorXi& ja, Eigen::VectorXd& a):
 mtype(-1), ia(ia), ja(ja), a(a)
 {}

 void PardisoSolver::set_type(int _mtype)
 {
   mtype = _mtype;
   init();
 }

 //template <typename IndexType, typename ScalarType>
 //void PardisoSolver<IndexType, ScalarType>::init()
 void PardisoSolver::init()
 {
   if (mtype ==-1)
   {
     printf("Pardiso mtype not set.");
     exit(1);
   }
   /* -------------------------------------------------------------------- */
   /* ..  Setup Pardiso control parameters.                                */
   /* -------------------------------------------------------------------- */

   error = 0;
   solver=0;/* use sparse direct solver */
   pardisoinit (pt,  &mtype, &solver, iparm, dparm, &error);

   if (error != 0)
   {
     if (error == -10 )
       printf("No license file found \n");
     if (error == -11 )
       printf("License is expired \n");
     if (error == -12 )
       printf("Wrong username or hostname \n");
     exit(1);
   }
   else
     printf("[PARDISO]: License check was successful ... \n");


   /* Numbers of processors, value of OMP_NUM_THREADS */
   var = getenv("OMP_NUM_THREADS");
   if(var != NULL)
     sscanf( var, "%d", &num_procs );
   else {
     printf("Set environment OMP_NUM_THREADS to 1");
     exit(1);
   }
   iparm[2]  = num_procs;
   //iparm[2]  = 1; // TODO: change me

   maxfct = 1;		/* Maximum number of numerical factorizations.  */
   mnum   = 1;         /* Which factorization to use. */

   msglvl = 0;         /* Print statistical information  */
   error  = 0;         /* Initialize error flag */


 //  /* -------------------------------------------------------------------- */
 //  /* .. Initialize the internal solver memory pointer. This is only */
 //  /* necessary for the FIRST call of the PARDISO solver. */
 //  /* -------------------------------------------------------------------- */
 //  for ( i = 0; i < 64; i++ )
 //  {
 //    pt[i] = 0;
 //  }


 }

 //template <typename IndexType, typename ScalarType>
 //void PardisoSolver<IndexType, ScalarType>::update_a(const std::vector<ScalarType> SS)
 void PardisoSolver::update_a(const std::vector<ScalarType> SS)
 {
   if (mtype ==-1)
   {
     printf("Pardiso mtype not set.");
     exit(1);
   }
   //for (int i=0; i<a.rows(); ++i) {
   for (int i=0; i<numUniqueElements; ++i) {
     a(i) = 0;
     for (int j=0; j<iis[i].size(); ++j)
       a(i) += SS[iis[i](j)];
   }
   
 }

 //template <typename IndexType, typename ScalarType>
 //void PardisoSolver<IndexType, ScalarType>::set_pattern(const std::vector<IndexType> &II,
 //                                const std::vector<IndexType> &JJ,
 //                                const std::vector<ScalarType> SS)
 void PardisoSolver::set_pattern(const std::vector<IndexType> &II,
                                                        const std::vector<IndexType> &JJ,
                                                        const std::vector<ScalarType> SS)


 {
   if (mtype ==-1)
   {
     printf("Pardiso mtype not set.");
     exit(1);
   }
   numRows = *(std::max_element(II.begin(), II.end()))+1;
   //assumption: we don't have i>j in II,JJ, hence these are not needed

 //  std::vector<int> pick;
 //  pick.reserve(II.size()/2);
 //  for (int i = 0; i<II.size();++i)
 //    if (II[i]<=JJ[i])
 //      pick.push_back(i);
 //  Eigen::MatrixXi M0(pick.size(),3);
 //  Eigen::VectorXd S1(pick.size(),1);
 //  for (int i = 0; i<pick.size();++i)
 //  {
 //    M0.row(i)<< II[pick[i]], JJ[pick[i]], i;
 //    S1[i] = SS[pick[i]];
 //  }

   //todo: make sure diagonal terms are included, even as zeros (pardiso claims this is necessary for best performance)
   Eigen::MatrixXi M0(II.size(),3);
   for (int i = 0; i<II.size();++i)
     M0.row(i)<< II[i], JJ[i], i;

   //temps
   Eigen::MatrixXi t;
   Eigen::VectorXi tI;

   Eigen::MatrixXi M_;
   igl::sortrows(M0, true, M_, tI);

   int si,ei,currI;
   si = 0;
   while (si<M_.rows())
   {
     currI = M_(si,0);
     ei = si;
     while (ei<M_.rows() && M_(ei,0) == currI)
       ++ei;
     igl::sortrows(M_.block(si, 1, ei-si, 2).eval(), true, t, tI);
     M_.block(si, 1, ei-si, 2) = t;
     si = ei+1;
   }

   Eigen::MatrixXi M;
   Eigen::VectorXi IM_;
   igl::unique_rows(M_.leftCols(2).eval(), M, IM_, tI);
   numUniqueElements = M.rows();
   iis.resize(numUniqueElements);
   for (int i=0; i<numUniqueElements; ++i)
   {
     si = IM_(i);
     if (i<numUniqueElements-1)
       ei = IM_(i+1);
     else
       ei = M_.rows();
     iis[i] = M_.block(si, 2, ei-si, 1);
   }

   a.resize(numUniqueElements, 1);
   for (int i=0; i<numUniqueElements; ++i)
   {
     a(i) = 0;
     for (int j=0; j<iis[i].size(); ++j)
 //      a(i) += S1(iis[i](j));
       a(i) += SS[iis[i](j)];
   }

   // now M_ and elements in sum have the row, column and indices in sum of the
   // unique non-zero elements in B1
   ia.setZero(numRows+1,1);ia(numRows) = numUniqueElements+1;
   ja = M.col(1).array()+1;
   currI = -1;
   for (int i=0; i<numUniqueElements; ++i)
   {
     if(currI != M(i,0))
     {
       ia(M(i,0)) = i+1;//do not subtract 1
       currI = M(i,0);
     }
   }

   // matrix in CRS can be expressed with ia, ja and iis

 }

 //template <typename IndexType, typename ScalarType>
 //void PardisoSolver<IndexType, ScalarType>::analyze_pattern()
 void PardisoSolver::analyze_pattern()
 {
   if (mtype ==-1)
   {
     printf("Pardiso mtype not set.");
     exit(1);
   }

 #ifdef PLOTS_PARDISO
   /* -------------------------------------------------------------------- */
   /*  .. pardiso_chk_matrix(...)                                          */
   /*     Checks the consistency of the given matrix.                      */
   /*     Use this functionality only for debugging purposes               */
   /* -------------------------------------------------------------------- */

   pardiso_chkmatrix  (&mtype, &numRows, a.data(), ia.data(), ja.data(), &error);
   if (error != 0) {
     printf("\nERROR in consistency of matrix: %d", error);
     exit(1);
   }
 #endif
   /* -------------------------------------------------------------------- */
   /* ..  Reordering and Symbolic Factorization.  This step also allocates */
   /*     all memory that is necessary for the factorization.              */
   /* -------------------------------------------------------------------- */
   phase = 11;

   pardiso (pt, &maxfct, &mnum, &mtype, &phase,
            &numRows, a.data(), ia.data(), ja.data(), &idum, &nrhs,
            iparm, &msglvl, &ddum, &ddum, &error, dparm);

   if (error != 0) {
     printf("\nERROR during symbolic factorization: %d", error);
     exit(1);
   }
 #ifdef PLOTS_PARDISO
   printf("\nReordering completed ... ");
   printf("\nNumber of nonzeros in factors  = %d", iparm[17]);
   printf("\nNumber of factorization MFLOPS = %d", iparm[18]);
 #endif

 }

 //template <typename IndexType, typename ScalarType>
 //bool PardisoSolver<IndexType, ScalarType>::factorize()
 bool PardisoSolver::factorize()
 {
   if (mtype ==-1)
   {
     printf("Pardiso mtype not set.");
     exit(1);
   }
   /* -------------------------------------------------------------------- */
   /* ..  Numerical factorization.                                         */
   /* -------------------------------------------------------------------- */
   phase = 22;
   //iparm[32] = 1; /* compute determinant */
   pardiso (pt, &maxfct, &mnum, &mtype, &phase,
            &numRows, a.data(), ia.data(), ja.data(), &idum, &nrhs,
            iparm, &msglvl, &ddum, &ddum, &error,  dparm);

   if (error != 0) {
     printf("\nERROR during numerical factorization: %d", error);
     exit(2);
   }
 #ifdef PLOTS_PARDISO
   printf ("\nFactorization completed ... ");
 #endif
   return (error ==0);
 }

 //template <typename IndexType, typename ScalarType>
 //void PardisoSolver<IndexType, ScalarType>::solve(const Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> &rhs,
 //                          Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> &result)
 void PardisoSolver::solve(Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> &rhs,
                                                  Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> &result)
 {
   if (mtype ==-1)
   {
     printf("Pardiso mtype not set.");
     exit(1);
   }

 #ifdef PLOTS_PARDISO
   /* -------------------------------------------------------------------- */
   /* ..  pardiso_chkvec(...)                                              */
   /*     Checks the given vectors for infinite and NaN values             */
   /*     Input parameters (see PARDISO user manual for a description):    */
   /*     Use this functionality only for debugging purposes               */
   /* -------------------------------------------------------------------- */

   pardiso_chkvec (&numRows, &nrhs, rhs.data(), &error);
   if (error != 0) {
     printf("\nERROR  in right hand side: %d", error);
     exit(1);
   }

   /* -------------------------------------------------------------------- */
   /* .. pardiso_printstats(...)                                           */
   /*    prints information on the matrix to STDOUT.                       */
   /*    Use this functionality only for debugging purposes                */
   /* -------------------------------------------------------------------- */

   pardiso_printstats (&mtype, &numRows, a.data(), ia.data(), ja.data(), &nrhs, rhs.data(), &error);
   if (error != 0) {
     printf("\nERROR right hand side: %d", error);
     exit(1);
   }

 #endif
   result.resize(numRows, 1);
   /* -------------------------------------------------------------------- */
   /* ..  Back substitution and iterative refinement.                      */
   /* -------------------------------------------------------------------- */
   phase = 33;

   iparm[7] = 1;       /* Max numbers of iterative refinement steps. */

   pardiso (pt, &maxfct, &mnum, &mtype, &phase,
            &numRows, a.data(), ia.data(), ja.data(), &idum, &nrhs,
            iparm, &msglvl, rhs.data(), result.data(), &error,  dparm);

   if (error != 0) {
     printf("\nERROR during solution: %d", error);
     exit(3);
   }
 #ifdef PLOTS_PARDISO
   printf("\nSolve completed ... ");
   printf("\nThe solution of the system is: ");
   for (i = 0; i < numRows; i++) {
     printf("\n x [%d] = % f", i, result.data()[i] );
   }
   printf ("\n\n");
 #endif
 }

 void PardisoSolver::free_numerical_factorization_memory() {
   phase = 0;                 /* Release internal memory. */

   pardiso (pt, &maxfct, &mnum, &mtype, &phase,
            &numRows, &ddum, ia.data(), ja.data(), &idum, &nrhs,
            iparm, &msglvl, &ddum, &ddum, &error,  dparm);
 }

 //template <typename IndexType, typename ScalarType>
 //PardisoSolver<IndexType, ScalarType>::~PardisoSolver()
 PardisoSolver::~PardisoSolver()
 {
   if (mtype == -1)
     return;
   /* -------------------------------------------------------------------- */
   /* ..  Termination and release of memory.                               */
   /* -------------------------------------------------------------------- */
   phase = -1;                 /* Release internal memory. */

   pardiso (pt, &maxfct, &mnum, &mtype, &phase,
            &numRows, &ddum, ia.data(), ja.data(), &idum, &nrhs,
            iparm, &msglvl, &ddum, &ddum, &error,  dparm);
 }
