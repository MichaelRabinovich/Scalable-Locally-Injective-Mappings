#ifndef LINE_SEARCH_PARAMETRIZER_H
#define LINE_SEARCH_PARAMETRIZER_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <map>
#include <set>
#include <vector>

#include "AlgorithmStage.h"
#include "Energy.h"
#include "Param_State.h"

class LinesearchParametrizer {

public:
  // does precomputation if it was not already done
  LinesearchParametrizer(Param_State* param_state);

  double parametrize( const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    Eigen::MatrixXd& cur_uv,
    Eigen::MatrixXd& dst_uv,
    Energy* energy,
    double cur_energy = -1);

  double compute_min_step_to_singularities(const Eigen::MatrixXd& uv,
                                            const Eigen::MatrixXi& F,
                                            Eigen::MatrixXd& d);
private:
  double line_search(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                              Eigen::MatrixXd& uv, const Eigen::MatrixXd& d,
                              double min_step_to_singularity, Energy* energy, double cur_energy);

  double get_min_pos_root_2D(const Eigen::MatrixXd& uv,const Eigen::MatrixXi& F,
            Eigen::MatrixXd& direc, int f);

  double get_min_pos_root_3D(const Eigen::MatrixXd& uv,const Eigen::MatrixXi& F,
            Eigen::MatrixXd& direc, int f);

  double get_smallest_pos_quad_zero(double a,double b, double c);
  int SolveP3(std::vector<double>& x,double a,double b,double c);

  bool bisection_wolfe_conditions_search(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Energy* e,
   const Eigen::MatrixXd& d, double min_step_to_singularity, Eigen::MatrixXd& uv, double old_e, 
   Eigen::MatrixXd& new_uv, double& new_e);

  double dot_prod_uv_format(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y);

  Param_State* m_state;

  const double wolfe_c1 = 1e-5;//1e-6;
  const double wolfe_c2 = 0.99;//0.999;
};

#endif // LINE_SEARCH_PARAMETRIZER_H
