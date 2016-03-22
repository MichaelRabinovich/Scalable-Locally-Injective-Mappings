#include "GlobalLocalParametrization.h"

#include "Param_State.h"
#include "eigen_stl_utils.h"
#include "parametrization_utils.h"
#include "LinesearchParametrizer.h"


#include <igl/boundary_loop.h>
#include <igl/cotmatrix.h>
#include <igl/edge_lengths.h>
#include <igl/local_basis.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/readOBJ.h>
#include <igl/repdiag.h>
#include <igl/vector_area_matrix.h>
#include <iostream>

#undef NDEBUG
#include <assert.h>
#define NDEBUG

using namespace std;

GlobalLocalParametrization::GlobalLocalParametrization(StateManager& state_manager, Param_State* m_state) : 
      ParametrizationAlgorithm(state_manager, m_state), WArap_p(NULL) {
  assert (m_state->F.cols() == 3);
  
  WArap_p = new LocalWeightedArapParametrizer(m_state);
}

void GlobalLocalParametrization::init_parametrization() {
  WArap_p->pre_calc();
  dirichlet_on_circle(m_state->V,m_state->F,m_state->uv);
  if (count_flips(m_state->V,m_state->F,m_state->uv) > 0) {
      //cout << "Cotan weights flattening has flips! Initializing with the positive cotan hack!" << endl;
      //dirichlet_on_circle_positive(m_state->V,m_state->F,m_state->uv);
      tutte_on_circle(m_state->V,m_state->F,m_state->uv);
  }
  m_state->energy = WArap_p->compute_energy(m_state->V, m_state->F, m_state->uv)/m_state->mesh_area;
}

void GlobalLocalParametrization::single_iteration() {
  single_line_search_arap();
  m_state->global_local_iters++;
}

void GlobalLocalParametrization::get_linesearch_params(Eigen::MatrixXd& dest_res,
                                                        Energy** param_energy) {
  dest_res = m_state->uv;
  WArap_p->parametrize(m_state->V,m_state->F, m_state->b,m_state->bc, dest_res);
  *param_energy = WArap_p;
}

void GlobalLocalParametrization::single_line_search_arap() {
  // weighted arap for riemannian metric
  LinesearchParametrizer linesearchParam(m_state);
  Eigen::MatrixXd dest_res;
  Energy* param_energy = NULL;
  m_state->timer.start();
  get_linesearch_params(dest_res, &param_energy);

  Eigen::MatrixXd old_uv = m_state->uv;
  //cout << "Copied old uv: " << m_state->timer.getElapsedTime() << endl;
  double old_energy = m_state->energy;

  m_state->energy = linesearchParam.parametrize(m_state->V,m_state->F, m_state->uv, dest_res, param_energy, m_state->energy*m_state->mesh_area)/m_state->mesh_area;
  cout << "Finished iter, time: " << m_state->timer.getElapsedTime() << endl;
}
