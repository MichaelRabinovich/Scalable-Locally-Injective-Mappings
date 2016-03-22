#ifndef __FAST_LS_BUILD_UTILS__H
#define __FAST_LS_BUILD_UTILS__H

#include <Eigen/Sparse>

#include <map>
#include <set>
#include <utility>
#include <vector>

// TODO: we need one mapping from K(vi,vj) to the big one, to the index, and from there we could create instruction lists by vi, vj offsets

//typedef Eigen::Tuple<int> inst;
struct instruction {
	int src1;
	int src2;
	int dst;

	bool operator < (const instruction& inst2) const {
        return (dst < inst2.dst);
    }
};

typedef std::vector< instruction > instruction_list;
typedef std::pair<int,int> fv_p;
typedef std::pair<int,int> vv_p;

//void add_dx_mult_dx_to_K(const Eigen::VectorXd& k1, const Eigen::VectorXd& k2, Eigen::VectorXd& K, const instruction_list& instructions);
void add_dx_mult_dx_to_K(const Eigen::VectorXd& k1, const Eigen::VectorXd& k2, Eigen::VectorXd& K, const instruction_list& instructions,
		const std::vector<int>& inst_idx);

void multiply_dx_by_W(const Eigen::VectorXd& k1, const Eigen::VectorXd& face_W, Eigen::VectorXd& res);
void multiply_dx_by_W_3d(const Eigen::VectorXd& k1, const Eigen::VectorXd& face_W, Eigen::VectorXd& res);

// gets dx (i.e FXV matrix)
void build_dx_k_maps(const int v_n, const Eigen::MatrixXi& F, Eigen::VectorXi& ai, Eigen::VectorXi& aj,
		instruction_list& instructions1, instruction_list& instructions2, instruction_list& instructions4,
		std::vector<int>& inst1_idx, std::vector<int>& inst2_idx, std::vector<int>& inst4_idx);

void build_dx_k_maps_3d(const int v_n, const Eigen::MatrixXi& F, Eigen::VectorXi& ai, Eigen::VectorXi& aj,
		instruction_list& instructions1, instruction_list& instructions2, instruction_list& instructions3,
		instruction_list& instructions5, instruction_list& instructions6, instruction_list& instructions9,
		std::vector<int>& inst1_idx, std::vector<int>& inst2_idx, std::vector<int>& inst3_idx,
		std::vector<int>& inst5_idx, std::vector<int>& inst6_idx, std::vector<int>& inst9_idx);

void build_index_from_instructions(instruction_list& instructions, std::vector<int>& inst_idx);
void dx_to_csr(const Eigen::SparseMatrix<double>& Dx, Eigen::VectorXi& ai, Eigen::VectorXi& aj, Eigen::VectorXd& a);
void csr_to_mat(int m, int n, const Eigen::VectorXi& ai, const Eigen::VectorXi& aj, const Eigen::VectorXd& a, Eigen::SparseMatrix<double>& A);

void tet_adjacency_list(const Eigen::MatrixXi& F, std::vector<std::vector<int> >& A);



#endif //__FAST_LS_BUILD_UTILS__H