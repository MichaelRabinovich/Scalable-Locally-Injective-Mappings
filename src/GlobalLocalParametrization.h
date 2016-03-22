#ifndef GLOBAL_LOCAL_PARAMETRIZATION_H
#define GLOBAL_LOCAL_PARAMETRIZATION_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <string>


#include "LocalWeightedArapParametrizer.h"
#include "ParametrizationAlgorithm.h"

#include <igl/jet.h>
#include <igl/readOBJ.h>
#include <igl/facet_components.h>
#include <igl/slice.h>
#ifdef HAS_GUI
#include <igl/viewer/Viewer.h>
#endif

class GlobalLocalParametrization : public ParametrizationAlgorithm {

public:

  GlobalLocalParametrization(StateManager& state_manager, Param_State* m_state);

  void init_parametrization();
  void single_iteration();

private:

  void single_line_search_arap();
  void get_linesearch_params(Eigen::MatrixXd& dest_res, Energy** param_energy);

  LocalWeightedArapParametrizer* WArap_p;
};

#endif // GLOBAL_LOCAL_PARAMETRIZATION_H
