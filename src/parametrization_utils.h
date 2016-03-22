#ifndef _PARAMETRIZATION_UTILS_H
#define _PARAMETRIZATION_UTILS_H

#include "igl/igl_inline.h"
#ifdef HAS_GUI
#include <igl/viewer/Viewer.h>
#endif

#include <igl/arap.h>
#include <Eigen/Core>
#include <set>
#include <tuple>
#include <vector>

using namespace std;

void compute_surface_gradient_matrix(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                                     const Eigen::MatrixXd& F1, const Eigen::MatrixXd& F2,
                                     Eigen::SparseMatrix<double>& D1, Eigen::SparseMatrix<double>& D2);

void compute_energies_with_jacobians(const Eigen::MatrixXd& V,
       const Eigen::MatrixXi& F, const Eigen::MatrixXd& Ji, Eigen::MatrixXd& uv, Eigen::VectorXd& areas,
       double& schaeffer_e, double& log_e, double& conf_e, double& norm_arap_e, double& amips, double& exp_symmd, double exp_factor, bool flips_linesearch = true);

void map_vertices_to_circle_area_normalized(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::VectorXi& bnd,
  Eigen::MatrixXd& UV);

void get_flips(const Eigen::MatrixXd& V,
               const Eigen::MatrixXi& F,
               const Eigen::MatrixXd& uv,
               std::vector<int>& flip_idx);

int count_flips(const Eigen::MatrixXd& V,
              const Eigen::MatrixXi& F,
              const Eigen::MatrixXd& uv);

void dirichlet_on_circle(const Eigen::MatrixXd& V,
              const Eigen::MatrixXi& F,
              Eigen::MatrixXd& uv);

bool tutte_on_circle(const Eigen::MatrixXd& V,
              const Eigen::MatrixXi& F,
              Eigen::MatrixXd& uv);

int get_euler_char(const Eigen::MatrixXd& V,
              const Eigen::MatrixXi& F);

#endif // _PARAMETRIZATION_UTILS_H
