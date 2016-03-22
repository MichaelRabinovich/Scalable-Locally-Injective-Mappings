#ifndef ALGORITHM_STAGE_H
#define ALGORITHM_STAGE_H

#ifdef HAS_GUI

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <map>
#include <set>
#include <vector>

#include <igl/viewer/Viewer.h>

#include "Param_State.h"

class AlgorithmStage {

public:
 AlgorithmStage(Param_State* dd_param) : m_dd(dd_param) {}
 virtual void key_pressed(igl::viewer::Viewer& viewer, unsigned char key) = 0;

 Param_State* m_dd;
};

#endif
#endif // ALGORITHM_STAGE_H
