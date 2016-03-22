#include "Param_State.h"

#include "igl/serialize.h"

using namespace std;

void Param_State::save(const std::string filename) {
   
   igl::serialize(V, "V", filename, true);
   igl::serialize(F,"F",filename);
   igl::serialize(M,"M",filename);
   igl::serialize(uv,"uv",filename);
   igl::serialize(v_num,"v_num",filename);
   igl::serialize(f_num,"f_num",filename);

   igl::serialize(mesh_area,"mesh_area",filename);
   igl::serialize(avg_edge_length,"avg_edge_length",filename);
   
   igl::serialize(flips_num,"flips_num", filename);
   igl::serialize(energy,"energy", filename);
   igl::serialize(global_symmds_energy,"symmds_energy",filename);
   igl::serialize(log_energy,"log_energy",filename);
   igl::serialize(conformal_energy,"conformal_energy",filename);   

   igl::serialize(flips_linesearch,"flips_linesearch",filename);

   igl::serialize(global_local_energy,"global_local_energy", filename);
   igl::serialize(global_local_iters, "global_local_iters", filename);
}

void Param_State::load(const std::string filename) {
   igl::deserialize(V,"V",filename);
   igl::deserialize(F,"F",filename);
   igl::deserialize(M,"M",filename);
   igl::deserialize(uv,"uv",filename);
   igl::deserialize(v_num,"v_num",filename);
   igl::deserialize(f_num,"f_num",filename);
   
   igl::deserialize(mesh_area,"mesh_area",filename);
   
   igl::deserialize(avg_edge_length,"avg_edge_length",filename);
   
   igl::deserialize(energy,"energy", filename);
   igl::deserialize(global_symmds_energy,"symmds_energy",filename);
   igl::deserialize(log_energy,"log_energy",filename);
   igl::deserialize(conformal_energy,"conformal_energy",filename);

   igl::deserialize(method,"method",filename);

   igl::deserialize(flips_linesearch,"flips_linesearch",filename);

   igl::deserialize(global_local_energy,"global_local_energy", filename);
   igl::deserialize(global_local_iters, "global_local_iters", filename);

   igl::deserialize(b,"b",filename);
   igl::deserialize(bc,"bc",filename);
}