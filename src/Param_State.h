#ifndef PARAM_STATE_H
#define PARAM_STATE_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <set>
#include <string>

#include "igl/arap.h"
#include "igl/Timer.h"

struct Param_State {

public:

  Param_State(){}

  void save(const std::string filename);
  void load(const std::string filename);

  // Global Information
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  Eigen::VectorXd M;
  Eigen::MatrixXd uv;

  int v_num;
  int f_num;

  // viewing params
  double mesh_area;
  double avg_edge_length;

  // result measurements
  int flips_num;
  double energy;
  double global_symmds_energy;
  double log_energy;
  double conformal_energy;
  double symmds_gradient_norm;
  double symmds_gradient_maxcoeff;
  double remeshing_energy;

  enum Param_Method {
    GLOBAL_ARAP_IRLS = 0,
    symmd,
    REMESHING,
    REMESHING_3D,
    BAR_DEFORMATION_3D,
    EQUALITY_CONSTRAINTS_DEMO,
    SEAMLESS_DEMO
  };
  Param_Method method;
  
  bool flips_linesearch;
  bool cnt_flips = false;

  enum GLOBAL_LOCAL_ENERGY {
  ARAP,
  LOG_ARAP,
  SYMMETRIC_DIRICHLET,
  CONFORMAL,
  EXP_CONFORMAL,
  AMIPS_ISO_2D,
  EXP_symmd
  };
  GLOBAL_LOCAL_ENERGY global_local_energy;
  int global_local_iters;

  enum GLOBAL_LOCAL_INIT {
  TUTTE,
  COTAN_WEIGHTS,
  LSCM,
  SCALING // mainly for testing purposes
  };
  GLOBAL_LOCAL_INIT global_local_init;

  enum GLOBAL_LOCAL_FACE_VISUALIZATION {
    NO_VIS = 0,
    WEIGHTS_VIS,
    ENERGY_DIR_VIS,
    ENERGY_DIR_SIGN_VIS,
    NORMAL_ENERGY_VIS,
  };
  GLOBAL_LOCAL_FACE_VISUALIZATION energy_vis = NO_VIS;

  // constraints (and or stiffness)
  Eigen::VectorXi b;
  Eigen::MatrixXd bc;
  double proximal_p;
  double soft_const_p;
  double exp_factor;
  
  // save line search samples
  igl::Timer timer;
  bool update_all_energies = false;
  enum UV_COLORING {
    NO_COLORING,
    DISTORTION_COLORING,
    NORMALS_COLORING,
    FLIPS_COLORING,
    WEIGHTS_COLORING
  };
  UV_COLORING uv_coloring;

  double min_color;
  double cube_deformation_size;
};

#endif // PARAM_STATE_H
