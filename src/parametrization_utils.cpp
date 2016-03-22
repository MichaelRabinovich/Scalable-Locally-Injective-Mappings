#include "parametrization_utils.h"

#include <igl/adjacency_matrix.h>
#include <igl/arap.h>
#include <igl/avg_edge_length.h>
#include <igl/boundary_loop.h>
#include <igl/colon.h>
#include <igl/harmonic.h>
#include <igl/edge_topology.h>
#include <igl/grad.h>
#include <igl/slice_into.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/lscm.h>
#include <igl/project_isometrically_to_plane.h>
#include <igl/doublearea.h>
#include <igl/volume.h>
#include <igl/per_face_normals.h>
#include <igl/writeOBJ.h>

#include <igl/project_isometrically_to_plane.h>
#include <igl/repdiag.h>
#include <igl/covariance_scatter_matrix.h>
#include <igl/edge_lengths.h>

#include "PardisoSolver.h"

#undef NDEBUG
#include <assert.h>
#define NDEBUG

void compute_surface_gradient_matrix(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                                     const Eigen::MatrixXd& F1, const Eigen::MatrixXd& F2,
                                     Eigen::SparseMatrix<double>& D1, Eigen::SparseMatrix<double>& D2) {
  using namespace Eigen;
  Eigen::SparseMatrix<double> G;
  //igl::grad(V,F,G);


  // Get grad
  const int fn = F.rows();  const int vn = V.rows();
  Eigen::MatrixXd grad3_3f(3, 3*fn);
  Eigen::MatrixXd fN; igl::per_face_normals(V,F,fN);
  Eigen::VectorXd Ar; igl::doublearea(V,F, Ar);
  for (int i = 0; i < fn; i++) {
     // renaming indices of vertices of triangles for convenience
    int i1 = F(i,0);
    int i2 = F(i,1);
    int i3 = F(i,2);

    // #F x 3 matrices of triangle edge vectors, named after opposite vertices
    Eigen::Matrix<double, 3,3> e;
    e.col(0) = V.row(i2) - V.row(i1);
    e.col(1) = V.row(i3) - V.row(i2);
    e.col(2) = V.row(i1) - V.row(i3);;
    
    Eigen::Matrix<double, 3,1> Fni = fN.row(i);
    double Ari = Ar(i);

    //grad3_3f(:,[3*i,3*i-2,3*i-1])=[0,-Fni(3), Fni(2);Fni(3),0,-Fni(1);-Fni(2),Fni(1),0]*e/(2*Ari);
    Eigen::Matrix<double, 3,3> n_M;
    n_M << 0,-Fni(2),Fni(1),Fni(2),0,-Fni(0),-Fni(1),Fni(0),0;
    Eigen::VectorXi R = igl::colon<int>(0,2);
    Eigen::VectorXi C(3); C  << 3*i+2,3*i,3*i+1;
    Eigen::MatrixXd res = (1./Ari)*(n_M*e);
    igl::slice_into(res,R,C,grad3_3f);
  }
  std::vector<Triplet<double> > Gx_trip,Gy_trip,Gz_trip;
  int val_idx = 0;
  for (int i = 0; i < fn; i++) {
    for (int j = 0; j < 3; j++) {
      Gx_trip.push_back(Triplet<double>(i, F(i,j), grad3_3f(0, val_idx)));
      Gy_trip.push_back(Triplet<double>(i, F(i,j), grad3_3f(1, val_idx)));
      Gz_trip.push_back(Triplet<double>(i, F(i,j), grad3_3f(2, val_idx)));
      val_idx++;
    }
  }
  SparseMatrix<double> Dx(fn,vn);  Dx.setFromTriplets(Gx_trip.begin(), Gx_trip.end());
  SparseMatrix<double> Dy(fn,vn);  Dy.setFromTriplets(Gy_trip.begin(), Gy_trip.end());
  SparseMatrix<double> Dz(fn,vn);  Dz.setFromTriplets(Gz_trip.begin(), Gz_trip.end());

  // This is good only if G = [Dx;Dy;Dz]. If not, then we should build the matrices in a different way
  // (I guess a for loop?)
  /*
  Eigen::SparseMatrix<double> Dx = G.block(0,0,F.rows(),G.cols());
  Eigen::SparseMatrix<double> Dy = G.block(F.rows(),0,F.rows(),G.cols());
  Eigen::SparseMatrix<double> Dz = G.block(2*F.rows(),0,F.rows(),G.cols());
  */
  // probably need Dx = G([0,3,6,...])
  /*
  cout << "G.row(0) = " << endl << G.row(0) << endl;
  cout << "G.row(1) = " << endl << G.row(1) << endl;
  cout << "G.rows() = " << G.rows() << " G.cols() = " << G.cols() << endl;
  Eigen::VectorXi R = igl::colon<int>(0,F.rows()-1);
  Eigen::VectorXi Rx = 3*R; Eigen::VectorXi Ry = 3*R.array() +1; Eigen::VectorXi Rz = 3*R.array() +2;
  Eigen::VectorXi C = igl::colon<int>(0,G.cols()-1);
  SparseMatrix<double> Dx;  igl::slice(G, Rx, C, Dx);
  SparseMatrix<double> Dy;  igl::slice(G, Ry, C, Dy);
  SparseMatrix<double> Dz;  igl::slice(G, Rz, C, Dz);
  */
  D1 = F1.col(0).asDiagonal()*Dx + F1.col(1).asDiagonal()*Dy + F1.col(2).asDiagonal()*Dz;
  D2 = F2.col(0).asDiagonal()*Dx + F2.col(1).asDiagonal()*Dy + F2.col(2).asDiagonal()*Dz;
}

void compute_energies_with_jacobians(const Eigen::MatrixXd& V,
       const Eigen::MatrixXi& F, const Eigen::MatrixXd& Ji, Eigen::MatrixXd& uv, Eigen::VectorXd& areas,
       double& schaeffer_e, double& log_e, double& conf_e, double& norm_arap_e, double& amips, double& exp_symmd, double exp_factor, bool flips_linesearch) {

  int f_n = F.rows();

  schaeffer_e = log_e = conf_e = 0; norm_arap_e = 0; amips = 0;
  Eigen::Matrix<double,2,2> ji;
  for (int i = 0; i < f_n; i++) {
    ji(0,0) = Ji(i,0); ji(0,1) = Ji(i,1);
    ji(1,0) = Ji(i,2); ji(1,1) = Ji(i,3);
    
    typedef Eigen::Matrix<double,2,2> Mat2;
    typedef Eigen::Matrix<double,2,1> Vec2;
    Mat2 ri,ti,ui,vi; Vec2 sing;
    igl::polar_svd(ji,ri,ti,ui,sing,vi);
    double s1 = sing(0); double s2 = sing(1);

    if (flips_linesearch) {
      schaeffer_e += areas(i) * (pow(s1,2) +pow(s1,-2) + pow(s2,2) + pow(s2,-2));
      log_e += areas(i) * (pow(log(s1),2) + pow(log(s2),2));
      double sigma_geo_avg = sqrt(s1*s2);
      //conf_e += areas(i) * (pow(log(s1/sigma_geo_avg),2) + pow(log(s2/sigma_geo_avg),2));
      conf_e += areas(i) * ( (pow(s1,2)+pow(s2,2))/(2*s1*s2) );
      norm_arap_e += areas(i) * (pow(s1-1,2) + pow(s2-1,2));
      amips +=  areas(i) * exp(exp_factor* (  0.5*( (s1/s2) +(s2/s1) ) + 0.25*( (s1*s2) + (1./(s1*s2)) )  ) );
      exp_symmd += areas(i) * exp(exp_factor*(pow(s1,2) +pow(s1,-2) + pow(s2,2) + pow(s2,-2)));
      //amips +=  areas(i) * exp(  0.5*( (s1/s2) +(s2/s1) ) + 0.25*( (s1*s2) + (1./(s1*s2)) )  ) ;
    } else {
      if (ui.determinant() * vi.determinant() > 0) {
        norm_arap_e += areas(i) * (pow(s1-1,2) + pow(s2-1,2));
      } else {
        // it is the distance form the flipped thing, this is slow, usefull only for debugging normal arap
        vi.col(1) *= -1;
        norm_arap_e += areas(i) * (ji-ui*vi.transpose()).squaredNorm();
      }
    }
    
  }

}

void map_vertices_to_circle_area_normalized(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F,
  const Eigen::VectorXi& bnd,
  Eigen::MatrixXd& UV) {
  
  Eigen::VectorXd dblArea_orig; // TODO: remove me later, waste of computations
  igl::doublearea(V,F, dblArea_orig);
  double area = dblArea_orig.sum()/2;
  double radius = sqrt(area / (M_PI));
  cout << "map_vertices_to_circle_area_normalized, area = " << area << " radius = " << radius << endl;

  // Get sorted list of boundary vertices
  std::vector<int> interior,map_ij;
  map_ij.resize(V.rows());
  interior.reserve(V.rows()-bnd.size());

  std::vector<bool> isOnBnd(V.rows(),false);
  for (int i = 0; i < bnd.size(); i++)
  {
    isOnBnd[bnd[i]] = true;
    map_ij[bnd[i]] = i;
  }

  for (int i = 0; i < (int)isOnBnd.size(); i++)
  {
    if (!isOnBnd[i])
    {
      map_ij[i] = interior.size();
      interior.push_back(i);
    }
  }

  // Map boundary to unit circle
  std::vector<double> len(bnd.size());
  len[0] = 0.;

  for (int i = 1; i < bnd.size(); i++)
  {
    len[i] = len[i-1] + (V.row(bnd[i-1]) - V.row(bnd[i])).norm();
  }
  double total_len = len[len.size()-1] + (V.row(bnd[0]) - V.row(bnd[bnd.size()-1])).norm();

  UV.resize(bnd.size(),2);
  for (int i = 0; i < bnd.size(); i++)
  {
    double frac = len[i] * (2. * M_PI) / total_len;
    UV.row(map_ij[bnd[i]]) << radius*cos(frac), radius*sin(frac);
  }

}

void get_flips(const Eigen::MatrixXd& V,
               const Eigen::MatrixXi& F,
               const Eigen::MatrixXd& uv,
               std::vector<int>& flip_idx) {
  flip_idx.resize(0);
  for (int i = 0; i < F.rows(); i++) {

    Eigen::Vector2d v1_n = uv.row(F(i,0)); Eigen::Vector2d v2_n = uv.row(F(i,1)); Eigen::Vector2d v3_n = uv.row(F(i,2));

    Eigen::MatrixXd T2_Homo(3,3);
    T2_Homo.col(0) << v1_n(0),v1_n(1),1;
    T2_Homo.col(1) << v2_n(0),v2_n(1),1;
    T2_Homo.col(2) << v3_n(0),v3_n(1),1;
    double det = T2_Homo.determinant();
    assert (det == det);
    if (det < 0) {
      //cout << "flip at face #" << i << " det = " << T2_Homo.determinant() << endl;
      flip_idx.push_back(i);
    }
  }
}
int count_flips(const Eigen::MatrixXd& V,
              const Eigen::MatrixXi& F,
              const Eigen::MatrixXd& uv) {

  std::vector<int> flip_idx;
  get_flips(V,F,uv,flip_idx);

  
  return flip_idx.size();
}

void dirichlet_on_circle(const Eigen::MatrixXd& V,
              const Eigen::MatrixXi& F,
              Eigen::MatrixXd& uv) {
       using namespace Eigen;
      typedef Matrix<double,Dynamic,1> VectorXS;
      
      // init (dirichlet)
      Eigen::VectorXi b;
      igl::boundary_loop(F,b);
      Eigen::MatrixXd bc;
      map_vertices_to_circle_area_normalized(V,F,b,bc);
      
      //igl::harmonic(V,F,bnd,bnd_uv,1,uv);

      SparseMatrix<double> L,M,Mi;
      igl::cotmatrix(V,F,L);
      SparseMatrix<double> Q = -L;
      uv.resize(V.rows(),bc.cols());

      #ifndef USE_PARDISO

      // Slow version (without Pardiso)
      const VectorXS B = VectorXS::Zero(V.rows(),1);
      igl::min_quad_with_fixed_data<double> data;
      igl::min_quad_with_fixed_precompute(Q,b,SparseMatrix<double>(),true,data);
      
      for(int w = 0;w<bc.cols();w++) {
        const VectorXS bcw = bc.col(w);
        VectorXS Ww;
        if(!igl::min_quad_with_fixed_solve(data,B,bcw,VectorXS(),Ww)) return;
        uv.col(w) = Ww;
      }

      #else
      
      int n = Q.rows(); int knowns = b.rows(); int unknowns = n-b.rows();
      std::vector<bool> unknown_mask;
      unknown_mask.resize(n,true);
      Eigen::VectorXi unknown(unknowns);
      for(int i = 0;i<knowns;i++) {
        unknown_mask[b(i)] = false;
      }
      int u = 0;
      for(int i = 0;i<n;i++) {
        if(unknown_mask[i]) {
          unknown(u) = i;
          u++;
        }
      }
      SparseMatrix<double> Quu; igl::slice(Q,unknown,unknown,Quu);
      SparseMatrix<double> Qub; igl::slice(Q,unknown,b,Qub);


      int nnz = Quu.nonZeros();
      std::vector<int> ii; std::vector<int> jj; std::vector<double> kk; ii.reserve(nnz); jj.reserve(nnz); kk.reserve(nnz);
      for (int k=0; k<Quu.outerSize(); ++k) {
          for (Eigen::SparseMatrix<double>::InnerIterator it(Quu,k); it; ++it) {
            if (it.row() <= it.col()) {
              ii.push_back(it.row()); jj.push_back(it.col()); kk.push_back(it.value());
            }
          }
      }
    
      Eigen::VectorXi ai,ji; Eigen::VectorXd a;
      PardisoSolver solver(ai,ji,a);
      solver.set_type(2);

      solver.set_pattern(ii,jj,kk);
      solver.analyze_pattern();
      solver.factorize();
  
      for(int w = 0;w<bc.cols();w++) {
        const VectorXS bcw = bc.col(w); VectorXS rhs = -Qub*bcw;
        VectorXS Ww;
        solver.solve(rhs,Ww);
        //uv.col(w) = Ww;
        int known_idx = 0; int unknown_idx = 0;
        for (int i = 0; i < n; i++) {
          if (unknown_mask[i]) {
            uv(i,w) = Ww(unknown_idx);
            unknown_idx++;
          }
        }
        for (int i = 0; i < b.rows(); i++) {
          uv(b(i),w) = bcw(i);
        }
        //cout << "known_idx = " << known_idx << " unknown_idx = " << unknown_idx << endl;
      }
      #endif
}

bool tutte_on_circle(const Eigen::MatrixXd& V,
              const Eigen::MatrixXi& F,
              Eigen::MatrixXd& uv) {
  using namespace Eigen;
  typedef Matrix<double,Dynamic,1> VectorXS;
// generate boundary conditions to a circle

  Eigen::SparseMatrix<double> A;
  igl::adjacency_matrix(F,A);

  Eigen::VectorXi b;
  igl::boundary_loop(F,b);
  Eigen::MatrixXd bc;
  map_vertices_to_circle_area_normalized(V,F,b,bc);

  
  // sum each row 
  Eigen::SparseVector<double> Asum;
  igl::sum(A,1,Asum);
  //Convert row sums into diagonal of sparse matrix
  Eigen::SparseMatrix<double> Adiag;
  igl::diag(Asum,Adiag);
  // Build uniform laplacian
  Eigen::SparseMatrix<double> Q;
  Q = Adiag - A;
  uv.resize(V.rows(),bc.cols());

  #ifndef USE_PARDISO
  // Slow version (without Pardiso)
  const Eigen::VectorXd B = Eigen::VectorXd::Zero(V.rows(),1);
  igl::min_quad_with_fixed_data<double> data;
  igl::min_quad_with_fixed_precompute(Q,b,Eigen::SparseMatrix<double>(),true,data);
  for(int w = 0;w<bc.cols();w++)
  {
    const Eigen::VectorXd bcw = bc.col(w);
    Eigen::VectorXd Ww;
    if(!igl::min_quad_with_fixed_solve(data,B,bcw,Eigen::VectorXd(),Ww))
    {
      return false;
    }
    uv.col(w) = Ww;
  }
  #else

  int n = Q.rows(); int knowns = b.rows(); int unknowns = n-b.rows();
  std::vector<bool> unknown_mask;
  unknown_mask.resize(n,true);
  Eigen::VectorXi unknown(unknowns);
  for(int i = 0;i<knowns;i++) {
    unknown_mask[b(i)] = false;
  }
  int u = 0;
  for(int i = 0;i<n;i++) {
    if(unknown_mask[i]) {
      unknown(u) = i;
      u++;
    }
  }
  SparseMatrix<double> Quu; igl::slice(Q,unknown,unknown,Quu);
  SparseMatrix<double> Qub; igl::slice(Q,unknown,b,Qub);


  int nnz = Quu.nonZeros();

  std::vector<int> ii; std::vector<int> jj; std::vector<double> kk; ii.reserve(nnz); jj.reserve(nnz); kk.reserve(nnz);
  for (int k=0; k<Quu.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(Quu,k); it; ++it) {
        if (it.row() <= it.col()) {
              ii.push_back(it.row()); jj.push_back(it.col()); kk.push_back(it.value());
        }
      }
  }

  Eigen::VectorXi ai,ji; Eigen::VectorXd a;
  PardisoSolver solver(ai,ji,a);
  solver.set_type(2);

  solver.set_pattern(ii,jj,kk);
  solver.analyze_pattern();
  solver.factorize();

  for(int w = 0;w<bc.cols();w++) {
    const VectorXS bcw = bc.col(w); VectorXS rhs = -Qub*bcw;
    VectorXS Ww;
    solver.solve(rhs,Ww);
    //uv.col(w) = Ww;
    int known_idx = 0; int unknown_idx = 0;
    for (int i = 0; i < n; i++) {
      if (unknown_mask[i]) {
        uv(i,w) = Ww(unknown_idx);
        unknown_idx++;
      }
    }
    for (int i = 0; i < b.rows(); i++) {
      uv(b(i),w) = bcw(i);
    }
    //cout << "known_idx = " << known_idx << " unknown_idx = " << unknown_idx << endl;
  }
  #endif
  return true;
}

int get_euler_char(const Eigen::MatrixXd& V,
              const Eigen::MatrixXi& F) {

  int euler_v = V.rows();
  Eigen::MatrixXi EV, FE, EF;
  igl::edge_topology(V, F, EV, FE, EF);
  int euler_e = EV.rows();
  int euler_f = F.rows();
    
  int euler_char = euler_v - euler_e + euler_f;
  return euler_char;
}
