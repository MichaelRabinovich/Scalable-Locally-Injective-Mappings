#ifndef LOCAL_WEIGHTED_ARAP_PARAMETRIZER_H
#define LOCAL_WEIGHTED_ARAP_PARAMETRIZER_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <map>
#include <set>
#include <vector>

#include "AlgorithmStage.h"
#include "SymmetricDirichlet.h"
#include "Energy.h"
#include "FastLsBuildUtils.h"
#include "Param_State.h"
#include "parametrization_utils.h"

#ifdef USE_PARDISO
#include "PardisoSolver.h"
#endif

#include "igl/arap.h"

class LocalWeightedArapParametrizer : public Energy {

public:
  LocalWeightedArapParametrizer(Param_State* state, bool remeshing = false);

  void parametrize( const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    Eigen::VectorXi& soft_b,
    Eigen::MatrixXd& soft_bc,
    Eigen::MatrixXd& uv);

  virtual double compute_energy(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                 Eigen::MatrixXd& uv);

  virtual void compute_grad(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                 Eigen::MatrixXd& uv, Eigen::MatrixXd& grad);

  void pre_calc();

  void compute_jacobians(const Eigen::MatrixXd& uv);

  Eigen::MatrixXd Ri,Ji;
  SymmetricDirichlet* symmd_p;
  Eigen::VectorXd W_11; Eigen::VectorXd W_12; Eigen::VectorXd W_21; Eigen::VectorXd W_22;
private:

  void update_weights_and_closest_rotations(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& uv);
  void solve_weighted_arap(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::MatrixXd& uv, Eigen::VectorXi& b,
      Eigen::MatrixXd& bc);

  void get_At_AtMA_fast();

  void add_proximal_penalty();
  
  Param_State* m_state;
  Eigen::VectorXd w11Dx,w12Dx,w11Dy,w12Dy,w21Dx,w22Dx,w21Dy,w22Dy;
  Eigen::VectorXd rhs;


  // Cached data for AtA and At matrix calculations
  Eigen::VectorXd a_x,a_y;
  Eigen::VectorXi dxi,dxj;
  Eigen::VectorXi ai,aj;
  Eigen::VectorXd K;
  instruction_list inst1,inst2,inst4;
  std::vector<int> inst1_idx,inst2_idx,inst4_idx;

  int f_n,v_n;

  #ifdef USE_PARDISO
  PardisoSolver m_solver;
  #endif

  bool first_solve;
  bool has_pre_calc = false;
};

#endif // #ifndef LOCAL_WEIGHTED_ARAP_PARAMETRIZER_H

