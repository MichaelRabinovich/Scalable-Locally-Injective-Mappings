#include "FastLsBuildUtils.h"

#include "igl/adjacency_list.h"
#include "igl/Timer.h"
#include "igl/unique.h"
#include "igl/vertex_triangle_adjacency.h"

#include "tbb/tbb.h"

using namespace std;

class FastATA_Multiplier {
    const Eigen::VectorXd& k1; const Eigen::VectorXd& k2; Eigen::VectorXd& K; const instruction_list& instructions;
	const std::vector<int>& inst_idx;
public:
	
    void operator()( const tbb::blocked_range<size_t>& r ) const {
        for( size_t i=r.begin(); i!=r.end(); ++i ) {
        	const int cur_idx = inst_idx[i]; const int next_idx = inst_idx[i+1];
			double val = 0;
			for (int j = 0; j < next_idx-cur_idx; j++) {
				val += k1[instructions[cur_idx+j].src1] * k2[instructions[cur_idx+j].src2];
			}
			K[instructions[cur_idx].dst] += val;
        }
    }
    FastATA_Multiplier( const Eigen::VectorXd& k1, const Eigen::VectorXd& k2, Eigen::VectorXd& K, const instruction_list& instructions,
					const std::vector<int>& inst_idx ) :
        k1(k1),k2(k2),K(K),instructions(instructions),inst_idx(inst_idx)
    {}
};

void add_dx_mult_dx_to_K(const Eigen::VectorXd& k1, const Eigen::VectorXd& k2, Eigen::VectorXd& K, const instruction_list& instructions,
		const std::vector<int>& inst_idx) {
	
	const int values_n = inst_idx.size() - 1;
	
	static tbb::affinity_partitioner ap;
	parallel_for(tbb::blocked_range<size_t>(0,values_n), FastATA_Multiplier(k1,k2,K,instructions,inst_idx),
		ap);
}

void build_dx_k_maps(const int v_n, const Eigen::MatrixXi& F, Eigen::VectorXi& ai, Eigen::VectorXi& aj,
							instruction_list& instructions1, instruction_list& instructions2, instruction_list& instructions4,
							std::vector<int>& inst1_idx, std::vector<int>& inst2_idx, std::vector<int>& inst4_idx) {
	using namespace std;
	using namespace Eigen;

	igl::Timer timer;
	timer.start();	

	const int f_n = F.rows();

	std::map<vv_p, int> vv_to_K;

	cout << "Started: " << timer.getElapsedTime() << endl;
	vector< vector<int> > V2V_tmp;
	igl::adjacency_list(F,V2V_tmp, false/*sorted*/);
	cout << "Built adjacency list: " << timer.getElapsedTime() << endl;
	// extending V2V to 2Vx2V, adj(2Vx2V) = [adj[V],adj[V]+v_n; adj[V],adj[V]+v_n];
	vector< vector<int> > V2V(V2V_tmp.size()*2);
	for (int vi = 0; vi < v_n; vi++) {
		V2V_tmp[vi].push_back(vi); // make sure we also have vi as a neighbour
		int v_nbd_size = V2V_tmp[vi].size();
		vector<int> nbs(v_nbd_size*2);
		 //cout << " vi = " << vi << endl;
		for (int j = 0; j < v_nbd_size; j++) {
			//cout << "nb j = " << V2V_tmp[vi][j] << endl;
			nbs[2*j] = V2V_tmp[vi][j];
			nbs[2*j+1] = V2V_tmp[vi][j] + v_n;
		}
		std::sort(nbs.begin(),nbs.end());
		V2V[vi] = V2V[vi + v_n] = nbs;
	}

	cout << "Built V2V: " << timer.getElapsedTime() << endl;
	// build vv_to_K (but we now look at K as a 2Vx2V sparse matrix)
	std::vector<int> ai_v,aj_v; ai_v.reserve(2*v_n); aj_v.reserve(12*v_n); ai_v.push_back(1);
	int K_idx = 0;
	for (int vi = 0; vi < 2*v_n; vi++) {
		int nb_v_num = V2V[vi].size();
		for (int vj_i = 0; vj_i < nb_v_num; vj_i++) {
			int vj = V2V[vi][vj_i];
			if (vj >= vi) {
				//cout << "vi = " << vi << " vj = " << vj << " K_idx = " << K_idx << endl;
				vv_to_K[vv_p(vi,vj)] = K_idx;
				aj_v.push_back(vj);
				K_idx++;
			}
		}
		ai_v.push_back(1+K_idx);
	}
	cout << "Built vv_to_K: " << timer.getElapsedTime() << endl;

	int ai_size = ai_v.size(); int aj_size = aj_v.size();
	
	ai.resize(ai_size); aj.resize(aj_size);
	for (int i = 0; i < ai_size; i++) {
		ai(i) = ai_v[i];
	}
	for (int i = 0; i < aj_size; i++) {
		aj(i) = aj_v[i]+1; //keep one based aj's for Pardiso
	}
	
	std::map<fv_p,int> fv_to_k; 
	
	vector< vector<int> > V2F; vector< vector<int> > V2Fi;
	igl::vertex_triangle_adjacency(v_n, F, V2F, V2Fi);
	cout << "Built V2F: " << timer.getElapsedTime() << endl;

	instructions1.reserve(3*3*f_n); instructions2.reserve(3*3*f_n); instructions4.reserve(3*3*f_n);
	for (int vi = 0; vi < v_n; vi++) {
		
		int f_nbd = V2F[vi].size();
		for (int f_i = 0; f_i < f_nbd; f_i++) {
			int cur_f = V2F[vi][f_i];
			int fvi_k = cur_f*3-1;
			if (vi >= F(cur_f,0)) fvi_k++;
			if (vi >= F(cur_f,1)) fvi_k++;
			if (vi >= F(cur_f,2)) fvi_k++;

			// go through all it's vertices
			for (int j = 0; j < 3; j++) {
				int vjf = F(cur_f,j);

				int fvj_k = cur_f*3-1;
				if (vjf >= F(cur_f,0)) fvj_k++;
				if (vjf >= F(cur_f,1)) fvj_k++;
				if (vjf >= F(cur_f,2)) fvj_k++;

				// now we have both of the indexes k1,k2 we need to multiply, but we have 3 different possible K index destination
				//	this is based on K1,K2,K4 indexes, so we add 3 instructions to the 3 lists (where K_mat = [K1,K2;K3,K4] is a symmetric matrix)
				int inst2_dst_K = vv_to_K[vv_p(vi,v_n+vjf)];
				//instructions2[fvi_k].push_back(instruction(fvj_k, inst2_dst_K));
				instructions2.push_back({fvi_k,fvj_k, inst2_dst_K});

				if (vi <= vjf) {
					int inst1_dst_K = vv_to_K[vv_p(vi,vjf)];
					instructions1.push_back({fvi_k,fvj_k, inst1_dst_K});
					//instructions1[fvi_k].push_back(instruction(fvj_k, inst1_dst_K));
					//cout << "vjf = " << vjf << " fvj_k = " << fvj_k << " inst1_dst_K = " << inst1_dst_K << endl;

					int inst4_dst_K = vv_to_K[vv_p(v_n+vi,v_n+vjf)];
					instructions4.push_back({fvi_k,fvj_k, inst4_dst_K});
					//instructions4[fvi_k].push_back(instruction(fvj_k, inst3_dst_K));
				}
			}	
		}
		
	}
	cout << "Built basic instructions: " << timer.getElapsedTime() << endl;
	std::sort(instructions1.begin(), instructions1.end());
	std::sort(instructions2.begin(), instructions2.end());
	std::sort(instructions4.begin(), instructions4.end());
	cout << "Sorted instructions: " << timer.getElapsedTime() << endl;

	// build the reference index
	inst2_idx.reserve(aj_size/2); // should be of size (2K)/4 = 0.5*K (since all is in)
	inst1_idx.reserve(aj_size/3); inst4_idx.reserve(aj_size/3); // should be of size (0.25*K -0.5*diag_size)

	build_index_from_instructions(instructions1, inst1_idx);	
	build_index_from_instructions(instructions2, inst2_idx);
	build_index_from_instructions(instructions4, inst4_idx);

	cout << "finished building instructions: " << timer.getElapsedTime() << endl;
}

void build_dx_k_maps_3d(const int v_n, const Eigen::MatrixXi& F, Eigen::VectorXi& ai, Eigen::VectorXi& aj,
		instruction_list& instructions1, instruction_list& instructions2, instruction_list& instructions3,
		instruction_list& instructions5, instruction_list& instructions6, instruction_list& instructions9,
		std::vector<int>& inst1_idx, std::vector<int>& inst2_idx, std::vector<int>& inst3_idx,
		std::vector<int>& inst5_idx, std::vector<int>& inst6_idx, std::vector<int>& inst9_idx) {
	using namespace std;
	using namespace Eigen;

	igl::Timer timer;
	timer.start();	

	const int f_n = F.rows();

	std::map<vv_p, int> vv_to_K;

	cout << "Started: " << timer.getElapsedTime() << endl;
	vector< vector<int> > V2V_tmp;
	tet_adjacency_list(F,V2V_tmp);
	cout << "Built adjacency list: " << timer.getElapsedTime() << endl;
	// extending V2V to 3Vx3V, adj(3Vx3V) = [adj[V],adj[V]+v_n,adj[V]+2*v_n; adj[V],adj[V]+v_n.adj[V]+2*v_n] (similar to the 2d function)
	vector< vector<int> > V2V(V2V_tmp.size()*3);
	for (int vi = 0; vi < v_n; vi++) {
		V2V_tmp[vi].push_back(vi); // make sure we also have vi as a neighbour
		int v_nbd_size = V2V_tmp[vi].size();
		vector<int> nbs(v_nbd_size*3);
		 //cout << " vi = " << vi << endl;
		for (int j = 0; j < v_nbd_size; j++) {
			nbs[3*j] = V2V_tmp[vi][j];
			nbs[3*j+1] = V2V_tmp[vi][j] + v_n;
			nbs[3*j+2] = V2V_tmp[vi][j] + 2*v_n;
		}
		std::sort(nbs.begin(),nbs.end());
		V2V[vi] = V2V[vi + v_n] = V2V[vi + 2*v_n] = nbs;
	}

	cout << "Built V2V: " << timer.getElapsedTime() << endl;
	// build vv_to_K (but we now look at K as a 2Vx2V sparse matrix)
	std::vector<int> ai_v,aj_v; ai_v.reserve(3*v_n); aj_v.reserve(18*v_n); ai_v.push_back(1);
	int K_idx = 0;
	for (int vi = 0; vi < 3*v_n; vi++) {
		int nb_v_num = V2V[vi].size();
		for (int vj_i = 0; vj_i < nb_v_num; vj_i++) {
			int vj = V2V[vi][vj_i];
			if (vj >= vi) {
				vv_to_K[vv_p(vi,vj)] = K_idx;
				aj_v.push_back(vj);
				K_idx++;
			}
		}
		ai_v.push_back(1+K_idx);
	}
	cout << "Built vv_to_K: " << timer.getElapsedTime() << endl;

	int ai_size = ai_v.size(); int aj_size = aj_v.size();
	
	ai.resize(ai_size); aj.resize(aj_size);
	for (int i = 0; i < ai_size; i++) {
		ai(i) = ai_v[i];
	}
	for (int i = 0; i < aj_size; i++) {
		aj(i) = aj_v[i]+1; //keep one based aj's for Pardiso
	}
	
	std::map<fv_p,int> fv_to_k; 
	
	vector< vector<int> > V2F; vector< vector<int> > V2Fi;
	igl::vertex_triangle_adjacency(v_n, F, V2F, V2Fi);
	cout << "Built V2F: " << timer.getElapsedTime() << endl;

	instructions1.reserve(6*4*f_n); instructions2.reserve(6*4*f_n); instructions3.reserve(6*4*f_n);
	instructions5.reserve(6*4*f_n); instructions6.reserve(6*4*f_n); instructions9.reserve(6*4*f_n);
	for (int vi = 0; vi < v_n; vi++) {
		
		int f_nbd = V2F[vi].size();
		for (int f_i = 0; f_i < f_nbd; f_i++) {
			int cur_f = V2F[vi][f_i];
			int fvi_k = cur_f*4-1;
			if (vi >= F(cur_f,0)) fvi_k++;
			if (vi >= F(cur_f,1)) fvi_k++;
			if (vi >= F(cur_f,2)) fvi_k++;
			if (vi >= F(cur_f,3)) fvi_k++;


			// go through all it's vertices
			for (int j = 0; j < 4; j++) {
				int vjf = F(cur_f,j);

				int fvj_k = cur_f*4-1;
				if (vjf >= F(cur_f,0)) fvj_k++;
				if (vjf >= F(cur_f,1)) fvj_k++;
				if (vjf >= F(cur_f,2)) fvj_k++;
				if (vjf >= F(cur_f,3)) fvj_k++;

				// now we have both of the indexes k1,k2 we need to multiply, but we have 6 different possible K index destination
				//	this is based on K1,K2,K3,K5,K6,K9 indexes, so we add 3 instructions to the 3 lists (where K_mat = [K1,K2,K3;K4,K5,K6;K7,K8,K9] is symmetric)
				int inst2_dst_K = vv_to_K[vv_p(vi,v_n+vjf)];
				instructions2.push_back({fvi_k,fvj_k, inst2_dst_K});

				int inst3_dst_K = vv_to_K[vv_p(vi,2*v_n+vjf)];
				instructions3.push_back({fvi_k,fvj_k, inst3_dst_K});

				int inst6_dst_K = vv_to_K[vv_p(v_n+vi,2*v_n+vjf)];
				instructions6.push_back({fvi_k,fvj_k, inst6_dst_K});

				if (vi <= vjf) {
					//diagonal entries
					int inst1_dst_K = vv_to_K[vv_p(vi,vjf)];
					instructions1.push_back({fvi_k,fvj_k, inst1_dst_K});

					int inst5_dst_K = vv_to_K[vv_p(v_n+vi,v_n+vjf)];
					instructions5.push_back({fvi_k,fvj_k, inst5_dst_K});

					int inst9_dst_K = vv_to_K[vv_p(2*v_n+vi,2*v_n+vjf)];
					instructions9.push_back({fvi_k,fvj_k, inst9_dst_K});

				}
			}	
		}
		
	}
	cout << "Built basic instructions: " << timer.getElapsedTime() << endl;
	std::sort(instructions1.begin(), instructions1.end());
	std::sort(instructions2.begin(), instructions2.end());
	std::sort(instructions3.begin(), instructions3.end());
	std::sort(instructions5.begin(), instructions5.end());
	std::sort(instructions6.begin(), instructions6.end());
	std::sort(instructions9.begin(), instructions9.end());
	cout << "Sorted instructions: " << timer.getElapsedTime() << endl;

	// build the reference index
	int full_num = ceil(2*aj_size/9.); int part_num = ceil(aj_size/9. + v_n);
	inst2_idx.reserve(full_num); inst3_idx.reserve(full_num); inst6_idx.reserve(full_num);
	inst1_idx.reserve(part_num); inst5_idx.reserve(part_num); inst9_idx.reserve(part_num);

	build_index_from_instructions(instructions1, inst1_idx);	
	build_index_from_instructions(instructions2, inst2_idx);
	build_index_from_instructions(instructions3, inst3_idx);
	build_index_from_instructions(instructions5, inst5_idx);
	build_index_from_instructions(instructions6, inst6_idx);
	build_index_from_instructions(instructions9, inst9_idx);

	cout << "finished building instructions: " << timer.getElapsedTime() << endl;
}

void build_index_from_instructions(instruction_list& instructions, std::vector<int>& inst_idx) {
	inst_idx.push_back(0); int cur_dst = instructions[0].dst;
	int i = 0;
	for (; i < instructions.size(); i++) {
		if (instructions[i].dst != cur_dst) {
			cur_dst = instructions[i].dst;
			inst_idx.push_back(i);
		}
	}
	inst_idx.push_back(i);
}

void multiply_dx_by_W(const Eigen::VectorXd& k1, const Eigen::VectorXd& face_W, Eigen::VectorXd& res) {
	if (res.rows() < k1.rows()){
		res.resize(k1.rows());
	}
	const int f_n = face_W.rows(); int idx = 0;
	for (int i = 0; i < f_n; i++) {
		double w = face_W(i);
		for (int j = 0; j < 3; j++) {
			res(idx) = w*k1(idx);
			idx++;
		}
	}
}

void multiply_dx_by_W_3d(const Eigen::VectorXd& k1, const Eigen::VectorXd& face_W, Eigen::VectorXd& res) {
	if (res.rows() < k1.rows()){
		res.resize(k1.rows());
	}
	const int f_n = face_W.rows(); int idx = 0;
	for (int i = 0; i < f_n; i++) {
		double w = face_W(i);
		for (int j = 0; j < 4; j++) {
			res(idx) = w*k1(idx);
			idx++;
		}
	}	
}

void dx_to_csr(const Eigen::SparseMatrix<double>& L, Eigen::VectorXi& ia, Eigen::VectorXi& ja, Eigen::VectorXd& a) {
   //assumption: we don't have i>j in II,JJ, hence these are not needed

 //  std::vector<int> pick;
 //  pick.reserve(II.size()/2);
 //  for (int i = 0; i<II.size();++i)
 //    if (II[i]<=JJ[i])
 //      pick.push_back(i);
 //  Eigen::MatrixXi M0(pick.size(),3);
 //  Eigen::VectorXd S1(pick.size(),1);
 //  for (int i = 0; i<pick.size();++i)
 //  {
 //    M0.row(i)<< II[pick[i]], JJ[pick[i]], i;
 //    S1[i] = SS[pick[i]];
 //  }

   //todo: make sure diagonal terms are included, even as zeros (pardiso claims this is necessary for best performance)

	int nnz = L.nonZeros();
    int expected_size = (nnz-L.rows())/2 + L.rows();
    std::vector<int> II; std::vector<int> JJ; std::vector<double> SS; II.reserve(expected_size); JJ.reserve(expected_size); SS.reserve(expected_size);
    for (int k=0; k<L.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(L,k); it; ++it) {
           II.push_back(it.row()); JJ.push_back(it.col()); SS.push_back(it.value());
        }
    }
   std::vector<Eigen::VectorXi> iis;
   int numRows = *(std::max_element(II.begin(), II.end()))+1;

   Eigen::MatrixXi M0(II.size(),3);
   for (int i = 0; i<II.size();++i)
     M0.row(i)<< II[i], JJ[i], i;

   //temps
   Eigen::MatrixXi t;
   Eigen::VectorXi tI;

   Eigen::MatrixXi M_;
   igl::sortrows(M0, true, M_, tI);

   int si,ei,currI;
   si = 0;
   while (si<M_.rows())
   {
     currI = M_(si,0);
     ei = si;
     while (ei<M_.rows() && M_(ei,0) == currI)
       ++ei;
     igl::sortrows(M_.block(si, 1, ei-si, 2).eval(), true, t, tI);
     M_.block(si, 1, ei-si, 2) = t;
     si = ei+1;
   }

   Eigen::MatrixXi M;
   Eigen::VectorXi IM_;
   igl::unique_rows(M_.leftCols(2).eval(), M, IM_, tI);
   int numUniqueElements = M.rows();
   iis.resize(numUniqueElements);
   for (int i=0; i<numUniqueElements; ++i)
   {
     si = IM_(i);
     if (i<numUniqueElements-1)
       ei = IM_(i+1);
     else
       ei = M_.rows();
     iis[i] = M_.block(si, 2, ei-si, 1);
   }

   a.resize(numUniqueElements, 1);
   for (int i=0; i<numUniqueElements; ++i)
   {
     a(i) = 0;
     for (int j=0; j<iis[i].size(); ++j)
 //      a(i) += S1(iis[i](j));
       a(i) += SS[iis[i](j)];
   }

   // now M_ and elements in sum have the row, column and indices in sum of the
   // unique non-zero elements in B1
   ia.setZero(numRows+1,1);ia(numRows) = numUniqueElements+1;
   ja = M.col(1).array()+1;
   currI = -1;
   for (int i=0; i<numUniqueElements; ++i)
   {
     if(currI != M(i,0))
     {
       ia(M(i,0)) = i+1;//do not subtract 1
       currI = M(i,0);
     }
   }
}
void csr_to_mat(int m, int n, const Eigen::VectorXi& ai, const Eigen::VectorXi& aj, const Eigen::VectorXd& a, Eigen::SparseMatrix<double>& A) {
	// Note: Pardiso's matrix j's are 1-based
	A = Eigen::SparseMatrix<double>(m,n);
	std::vector<Eigen::Triplet<double> > IJV;
  	IJV.reserve(a.rows());
	int k_idx = 0;
	for (int i = 0; i < m; i++) {
		int nnz_in_line = ai[i+1] - ai[i];
		for (int j = 0; j < nnz_in_line; j++) {
			IJV.push_back(Eigen::Triplet<double>(i,aj[k_idx]-1,a[k_idx])); // the j are 1-based!
			k_idx++;
		}
	}
	A.setFromTriplets(IJV.begin(),IJV.end());
}

void tet_adjacency_list(const Eigen::MatrixXi& F, std::vector<std::vector<int> >& A) {
  A.clear(); 
  A.resize(F.maxCoeff()+1);
  cout << "A.size() = " << A.size() << endl;
  
  Eigen::Matrix<int,Eigen::Dynamic,2> edges;
  edges.resize(6,2);
    edges << 
      1,2,
      2,0,
      0,1,
      3,0,
      3,1,
      3,2;

  // Loop over faces
  for(int i = 0;i<F.rows();i++)
  {
 	// loop over edges of element
    for(int e = 0;e<edges.rows();e++) {
      int v1 = F(i,edges(e,0));
      int v2 = F(i,edges(e,1));
      //cout << "i = " << i << " j = " << j << endl;
      A[v1].push_back(v2);
      A[v2].push_back(v1);
    }
  }
  // Remove duplicates
  for(int i=0; i<(int)A.size();++i) // TODO: probably redundant
  {
    std::sort(A[i].begin(), A[i].end());
    A[i].erase(std::unique(A[i].begin(), A[i].end()), A[i].end());
  }
  
}