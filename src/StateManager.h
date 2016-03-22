#ifndef STATE_MANAGER_H
#define STATE_MANAGER_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <map>
#include <set>
#include <vector>

#ifdef HAS_GUI
#include <igl/viewer/Viewer.h>
#endif

#include "Param_State.h"

class StateManager {

public:
 StateManager() : cur_state(0) {};

 void add_state(Param_State param_state);
 void save(std::string& path);
 void load(std::string& path);
 void reset();
 int states_num() { return states.size();};
 Param_State* get_last_state() { return &(states[states.size()-1]);};
 Param_State* get_cur_state() { return &(states[cur_state]);};
 Param_State* get_state(int idx) { return &(states[idx]);};

 void prev_state();
 void next_state();

 int cur_state;
 std::vector<Param_State> states;
};

#endif // STATE_MANAGER_H
