#include "eigen_stl_utils.h"

void int_set_to_eigen_vector(const std::set<int>& int_set, Eigen::VectorXi& vec) {
	vec.resize(int_set.size()); int idx = 0;
	for(auto f : int_set) {
  		vec(idx) = f; idx++;
    }
}
void double_vector_to_eigen_vector(const std::vector<double>& double_vec, Eigen::VectorXd& vec) {
	vec.resize(double_vec.size()); int idx = 0;
	for(auto f : double_vec) {
  		vec(idx) = f; idx++;
    }
}

void eigen_vector_to_int_set(const Eigen::VectorXi& vec, std::set<int>& int_set) {
	for (int i = 0; i < vec.rows(); i++) {
		int_set.insert(vec(i));
	}
}

void eigen_mat_to_double_tuple(const Eigen::MatrixXd& mat, vector< tuple<double,double,double> > tup) {
	tup.resize(mat.rows());
	for (int i = 0; i < mat.rows(); i++) {
		tup[i] = tuple<double,double,double>(mat(i,0),mat(i,1),mat(i,2));
	}
}


void mat2_to_vec(const Eigen::MatrixXd& mat, Eigen::VectorXd& vec) {
	// x1,y1,x2,y2,...
	Eigen::MatrixXd tmp = mat.transpose();
	tmp.resize(mat.cols()*mat.rows(),1);
	vec = tmp;
}

void vec_to_mat2(const Eigen::VectorXd& vec, Eigen::MatrixXd& mat) {
	mat.resize(vec.rows()/2, 2);
	for (int i = 0; i < mat.rows(); i++) {
		mat.row(i) << vec(i*2), vec(i*2+1);
	}
}