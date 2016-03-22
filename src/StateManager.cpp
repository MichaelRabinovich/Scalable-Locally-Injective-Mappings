#include "StateManager.h"

#include "igl/serialize.h"

using namespace std;

void StateManager::add_state(Param_State param_state) {
	states.push_back(param_state);
}

void StateManager::save(std::string& filename) {
	int states_num = states.size();
	igl::serialize(states_num,"states_num",filename, true);
	cout << "#states = " << states_num << endl;
	for (int i = 0; i < states_num; i++) {
		std::string state_f = filename + std::string ("-") + std::to_string(i);
		states[i].save(state_f);
	}
	cur_state = states_num - 1; // 0 indexing
}

void StateManager::load(std::string& filename) {
	cout << "StateManager::load() " << endl;
	reset();

	int states_num;
	igl::deserialize(states_num,"states_num",filename);
	cout << "#states = " << states_num << endl;
	states.resize(states_num);
	for (int i = 0; i < states_num; i++) {
		std::string state_f = filename + std::string ("-") + std::to_string(i);
		cout << "loading state " << i << endl;
		try {
			states[i].load(state_f);
		} catch (...) {

		}
	}
	cur_state = states_num - 1;
}

void StateManager::reset() { 
	states = std::vector<Param_State>();
	cur_state = 0;
}

void StateManager::prev_state() {
 	if (cur_state > 0) cur_state--;
 	cout << "current state = " << cur_state << endl;
 }

 void StateManager::next_state() {
 	int states_num = states.size();
 	if (cur_state < (states_num -1)) cur_state++;
 	cout << "current state = " << cur_state << endl;
 }