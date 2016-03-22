#ifndef _EIGEN_STL_UTILS_H__
#define _EIGEN_STL_UTILS_H__

#include <Eigen/Core>
#include <set>
#include <tuple>
#include <vector>

using namespace std;

void int_set_to_eigen_vector(const std::set<int>& int_set, Eigen::VectorXi& vec);
void double_vector_to_eigen_vector(const std::vector<double>& double_vec, Eigen::VectorXd& vec);
void eigen_vector_to_int_set(const Eigen::VectorXi& vec, std::set<int>& int_set);
void eigen_mat_to_double_tuple(const Eigen::MatrixXd& mat, vector< tuple<double,double,double> > tup);

void mat2_to_vec(const Eigen::MatrixXd& mat, Eigen::VectorXd& vec);
void vec_to_mat2(const Eigen::VectorXd& vec, Eigen::MatrixXd& mat);

#endif // _EIGEN_STL_UTILS_H__
