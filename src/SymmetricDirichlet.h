#ifndef SYMMETRIC_DIRICHLET_H
#define SYMMETRIC_DIRICHLET_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <map>
#include <set>
#include <vector>

#include "AlgorithmStage.h"
#include "Param_State.h"

class SymmetricDirichlet {

public:
  // does precomputation if it was not already done
  SymmetricDirichlet(Param_State* dd_param, std::vector<int> b = std::vector<int>());

  void parametrize( const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                        Eigen::MatrixXd& uv);

  double parametrize_LBFGS( const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                        Eigen::MatrixXd& uv, int max_iter, int dummy);
  double parametrize_LBFGS( const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                        Eigen::MatrixXd& uv, int max_iter = 10000);

  double one_step( const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                        Eigen::MatrixXd& uv);

  double compute_energy(const Eigen::MatrixXd& V,
             const Eigen::MatrixXi& F,
             const Eigen::MatrixXd& uv);

  void compute_negative_gradient(const Eigen::MatrixXd& V,
          const Eigen::MatrixXi& F,
          const Eigen::MatrixXd& uv,
          Eigen::MatrixXd& neg_grad);

  Param_State* m_dd;
  double cur_energy;
  double cur_riemann_energy;
  bool has_converged;

  // cached computations
  std::vector<double> m_cached_l_energy_per_face;
  std::vector<double> m_cached_r_energy_per_face;

private:

  double single_gradient_descent( const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                        Eigen::MatrixXd& uv);

  double LineSearch_michael_armijo_imp(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                    Eigen::MatrixXd& uv, const Eigen::MatrixXd& d, double max_step_size);

  // https://github.com/PatWie/CppNumericalSolvers/
  double LineSearch_patwie_armijo_imp(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                    Eigen::MatrixXd& uv, const Eigen::MatrixXd& grad,const Eigen::MatrixXd& d, double max_step_size);

  double compute_energy(const Eigen::MatrixXd& V,
             const Eigen::MatrixXi& F,
             const Eigen::VectorXd& uv);


void compute_negative_gradient(const Eigen::MatrixXd& V,
            const Eigen::MatrixXi& F,
            const Eigen::VectorXd& uv_vec,
            Eigen::VectorXd& neg_grad_vec);

  double compute_max_step_from_singularities(const Eigen::MatrixXd& uv,
                                            const Eigen::MatrixXi& F,
                                            Eigen::MatrixXd& grad);

  double get_min_pos_root(const Eigen::MatrixXd& uv,const Eigen::MatrixXi& F,
                        Eigen::MatrixXd& d, int f);

  double compute_face_energy_left_part(const Eigen::MatrixXd& V,
             const Eigen::MatrixXi& F,
             const Eigen::MatrixXd& uv, int f_idx);

  double compute_face_energy_right_part(const Eigen::MatrixXd& V,
             const Eigen::MatrixXi& F,
             const Eigen::MatrixXd& uv,int f_idx,
             double orig_t_dbl_area);

  double compute_face_energy_part(const Eigen::MatrixXd& V,
                     const Eigen::MatrixXi& F,
                     const Eigen::MatrixXd& uv,
                     bool is_left_grad);

  bool check_grad(const Eigen::MatrixXd& V,
                     const Eigen::MatrixXi& F,
                     const Eigen::MatrixXd& uv, int v_idx, Eigen::RowVector2d grad,
                     bool is_left_grad) ;

  void precompute(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);

  void zero_out_const_vertices_search_direction(Eigen::MatrixXd& d);

  void update_results(double new_energy);

  long m_iter;
  bool has_precomputed;

  // cached computations
  std::vector<double> m_cached_edges_1;
  std::vector<double> m_cached_edges_2;
  std::vector<double> m_cached_dot_prod;
  Eigen::MatrixXd m_cot_entries;
  Eigen::VectorXd m_dblArea_orig;
  Eigen::VectorXd m_dbl_sqrt_area;

  std::vector<int> m_b; // constrained vertices
  int m_arap_iter;
};

#endif // SYMMETRIC_DIRICHLET_H
