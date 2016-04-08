#include "LinesearchParametrizer.h"

#include "eigen_stl_utils.h"
#include "parametrization_utils.h"

#include "igl/avg_edge_length.h"

#undef NDEBUG
#include <assert.h>
#define NDEBUG

LinesearchParametrizer::LinesearchParametrizer (Param_State* param_state) : m_state(param_state) {
	// empty
}

double LinesearchParametrizer::parametrize( const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
    Eigen::MatrixXd& uv, Eigen::MatrixXd& dst_uv, Energy* energy, double cur_energy) {

    Eigen::MatrixXd d = dst_uv - uv;

    double min_step_to_singularity = compute_min_step_to_singularities(uv,F,d);

    return line_search(V,F,uv,d,min_step_to_singularity, energy, cur_energy);
}
double LinesearchParametrizer::line_search(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
                              Eigen::MatrixXd& uv, const Eigen::MatrixXd& d, 
                              double min_step_to_singularity, Energy* energy, double cur_energy) {
  double old_energy;
  if (cur_energy > 0) {
    old_energy = cur_energy;  
  } else {
    old_energy = energy->compute_energy(V,F,uv); // no energy was given -> need to compute the current energy
  }
  double new_energy = old_energy;
  
  double linesearch_e; Eigen::MatrixXd linesearch_uv;
  bool ret = bisection_wolfe_conditions_search(V,F, energy, d, min_step_to_singularity, uv, old_energy,
             linesearch_uv, linesearch_e);

  if (ret) {
    uv = linesearch_uv;
    new_energy = linesearch_e;
  } else {
    cout << "Error: linesearch cannot progress anymore" << endl;
  }
  return new_energy;
}

bool LinesearchParametrizer::bisection_wolfe_conditions_search(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Energy* e,
   const Eigen::MatrixXd& d, double min_step_to_singularity, Eigen::MatrixXd& uv, double old_e, 
   Eigen::MatrixXd& new_uv, double& new_e) {

  // first find maximum direction
  double max_alpha = 0.99 * min_step_to_singularity;
  double min_alpha = 0;
  double step_size = min(1., 0.8*min_step_to_singularity);

  new_uv = uv + step_size * d;
  Eigen::MatrixXd old_grad(uv.rows(), uv.cols()); old_grad.setZero();
  e->compute_grad(V,F,uv,old_grad);
  double d_ograd_prod = dot_prod_uv_format(d,old_grad);
  
  new_e = e->compute_energy(V,F,new_uv);

   const int MAX_ITER = 50;
   int iters = 0;
   for (; iters < MAX_ITER; iters++) {

      // sufficient decrease check: Wolfe condition number 1
      if (new_e > old_e + wolfe_c1 * step_size * d_ograd_prod) {
        max_alpha = step_size;
        step_size = (min_alpha + max_alpha) / 2.;

      } else {
        
        Eigen::MatrixXd new_grad(uv.rows(), uv.cols()); new_grad.setZero();
        e->compute_grad(V,F,new_uv,new_grad);
        double d_ngrad_prod = dot_prod_uv_format(d,new_grad);
        
        // curvature condition: (Strong) Wolfe Condition number 2
        if (abs(d_ngrad_prod) > wolfe_c2*abs(d_ograd_prod)) {
          if (d_ngrad_prod > 0) {
            // make me smaller
            max_alpha = step_size;            
          } else {
            // make me bigger
            min_alpha = step_size;
          }
          step_size = (min_alpha + max_alpha) / 2.;
        } else {
          return true;
        }
    }
    new_uv = uv + step_size * d;
    new_e = e->compute_energy(V,F,new_uv);
  }
   return (iters < MAX_ITER); 
}

double LinesearchParametrizer::dot_prod_uv_format(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) {
  assert(x.rows() == y.rows());
  assert(x.cols() == y.cols());
  assert(x.cols() == 2); // uv dot product

  double dp = 0;
  for (int i = 0; i < x.rows(); i++) {
    dp += x(i,0)*y(i,0) + x(i,1)*y(i,1);
  }
  return dp;
}

 double LinesearchParametrizer::compute_min_step_to_singularities(const Eigen::MatrixXd& uv,
                                            const Eigen::MatrixXi& F,
                                            Eigen::MatrixXd& d) {
    double max_step = INFINITY;

    // The if statement is outside the for loops to avoid branching/ease parallelizing
    if (uv.cols() == 2) {
      for (int f = 0; f < F.rows(); f++) {
        double min_positive_root = get_min_pos_root_2D(uv,F,d,f);
        max_step = min(max_step, min_positive_root);
      }
    } else { // volumetric deformation
      for (int f = 0; f < F.rows(); f++) {
        double min_positive_root = get_min_pos_root_3D(uv,F,d,f);
        max_step = min(max_step, min_positive_root);
      }
    }
    return max_step;
 }

 double LinesearchParametrizer::get_min_pos_root_2D(const Eigen::MatrixXd& uv,const Eigen::MatrixXi& F,
            Eigen::MatrixXd& d, int f) {
/*
      Symbolic matlab for equation 4 at the paper (this is how to recreate the formulas below)
      U11 = sym('U11');
      U12 = sym('U12');
      U21 = sym('U21');
      U22 = sym('U22');
      U31 = sym('U31');
      U32 = sym('U32');

      V11 = sym('V11');
      V12 = sym('V12');
      V21 = sym('V21');
      V22 = sym('V22');
      V31 = sym('V31');
      V32 = sym('V32');

      t = sym('t');

      U1 = [U11,U12];
      U2 = [U21,U22];
      U3 = [U31,U32];

      V1 = [V11,V12];
      V2 = [V21,V22];
      V3 = [V31,V32];

      A = [(U2+V2*t) - (U1+ V1*t)];
      B = [(U3+V3*t) - (U1+ V1*t)];
      C = [A;B];

      solve(det(C), t);
      cf = coeffs(det(C),t); % Now cf(1),cf(2),cf(3) holds the coefficients for the polynom. at order c,b,a
    */

  int v1 = F(f,0); int v2 = F(f,1); int v3 = F(f,2);
  // get quadratic coefficients (ax^2 + b^x + c)
  #define U11 uv(v1,0)
  #define U12 uv(v1,1)
  #define U21 uv(v2,0)
  #define U22 uv(v2,1)
  #define U31 uv(v3,0)
  #define U32 uv(v3,1)

  #define V11 d(v1,0)
  #define V12 d(v1,1)
  #define V21 d(v2,0)
  #define V22 d(v2,1)
  #define V31 d(v3,0)
  #define V32 d(v3,1)
  
  
  double a = V11*V22 - V12*V21 - V11*V32 + V12*V31 + V21*V32 - V22*V31;
  double b = U11*V22 - U12*V21 - U21*V12 + U22*V11 - U11*V32 + U12*V31 + U31*V12 - U32*V11 + U21*V32 - U22*V31 - U31*V22 + U32*V21;
  double c = U11*U22 - U12*U21 - U11*U32 + U12*U31 + U21*U32 - U22*U31;

  return get_smallest_pos_quad_zero(a,b,c);
}

double LinesearchParametrizer::get_smallest_pos_quad_zero(double a,double b, double c) {
  double t1,t2;
  if (a != 0) {
    double delta_in = pow(b,2) - 4*a*c;
    if (delta_in < 0) {
      return INFINITY;
    }
    double delta = sqrt(delta_in);
    t1 = (-b + delta)/ (2*a);
    t2 = (-b - delta)/ (2*a);
  } else {
    t1 = t2 = -b/c;
  }
  assert (std::isfinite(t1));
  assert (std::isfinite(t2));

  double tmp_n = min(t1,t2);
  t1 = max(t1,t2); t2 = tmp_n;
  if (t1 == t2) {
    return INFINITY; // means the orientation flips twice = doesn't flip?
  }
  // return the smallest negative root if it exists, otherwise return infinity
  if (t1 > 0) {
    if (t2 > 0) {
      return t2;
    } else {
      return t1;
    }
  } else {
    return INFINITY;
  }
}

double LinesearchParametrizer::get_min_pos_root_3D(const Eigen::MatrixXd& uv,const Eigen::MatrixXi& F,
            Eigen::MatrixXd& direc, int f) {
  /*
        +-1/6 * |ax ay az 1|
                |bx by bz 1|
                |cx cy cz 1|
                |dx dy dz 1|
      Every point ax,ay,az has a search direction a_dx,a_dy,a_dz, and so we add those to the matrix, and solve the cubic to find the step size t for a 0 volume
      Symbolic matlab:
        syms a_x a_y a_z a_dx a_dy a_dz % tetrahedera point and search direction
        syms b_x b_y b_z b_dx b_dy b_dz
        syms c_x c_y c_z c_dx c_dy c_dz
        syms d_x d_y d_z d_dx d_dy d_dz
        syms t % Timestep var, this is what were looking for


        a_plus_t = [a_x,a_y,a_z] + t*[a_dx,a_dy,a_dz];
        b_plus_t = [b_x,b_y,b_z] + t*[b_dx,b_dy,b_dz];
        c_plus_t = [c_x,c_y,c_z] + t*[c_dx,c_dy,c_dz];
        d_plus_t = [d_x,d_y,d_z] + t*[d_dx,d_dy,d_dz];

        vol_mat = [a_plus_t,1;b_plus_t,1;c_plus_t,1;d_plus_t,1]
        //cf = coeffs(det(vol_det),t); % Now cf(1),cf(2),cf(3),cf(4) holds the coefficients for the polynom
        [coefficients,terms] = coeffs(det(vol_det),t); % terms = [ t^3, t^2, t, 1], Coefficients hold the coeff we seek
  */
  int v1 = F(f,0); int v2 = F(f,1); int v3 = F(f,2); int v4 = F(f,3);
  #define a_x uv(v1,0)
  #define a_y uv(v1,1)
  #define a_z uv(v1,2)
  #define b_x uv(v2,0)
  #define b_y uv(v2,1)
  #define b_z uv(v2,2)
  #define c_x uv(v3,0)
  #define c_y uv(v3,1)
  #define c_z uv(v3,2)
  #define d_x uv(v4,0)
  #define d_y uv(v4,1)
  #define d_z uv(v4,2)

  #define a_dx direc(v1,0)
  #define a_dy direc(v1,1)
  #define a_dz direc(v1,2)
  #define b_dx direc(v2,0)
  #define b_dy direc(v2,1)
  #define b_dz direc(v2,2)
  #define c_dx direc(v3,0)
  #define c_dy direc(v3,1)
  #define c_dz direc(v3,2)
  #define d_dx direc(v4,0)
  #define d_dy direc(v4,1)
  #define d_dz direc(v4,2)

  // Find solution for: a*t^3 + b*t^2 + c*d +d = 0
  double a = a_dx*b_dy*c_dz - a_dx*b_dz*c_dy - a_dy*b_dx*c_dz + a_dy*b_dz*c_dx + a_dz*b_dx*c_dy - a_dz*b_dy*c_dx - a_dx*b_dy*d_dz + a_dx*b_dz*d_dy + a_dy*b_dx*d_dz - a_dy*b_dz*d_dx - a_dz*b_dx*d_dy + a_dz*b_dy*d_dx + a_dx*c_dy*d_dz - a_dx*c_dz*d_dy - a_dy*c_dx*d_dz + a_dy*c_dz*d_dx + a_dz*c_dx*d_dy - a_dz*c_dy*d_dx - b_dx*c_dy*d_dz + b_dx*c_dz*d_dy + b_dy*c_dx*d_dz - b_dy*c_dz*d_dx - b_dz*c_dx*d_dy + b_dz*c_dy*d_dx;
  double b = a_dy*b_dz*c_x - a_dy*b_x*c_dz - a_dz*b_dy*c_x + a_dz*b_x*c_dy + a_x*b_dy*c_dz - a_x*b_dz*c_dy - a_dx*b_dz*c_y + a_dx*b_y*c_dz + a_dz*b_dx*c_y - a_dz*b_y*c_dx - a_y*b_dx*c_dz + a_y*b_dz*c_dx + a_dx*b_dy*c_z - a_dx*b_z*c_dy - a_dy*b_dx*c_z + a_dy*b_z*c_dx + a_z*b_dx*c_dy - a_z*b_dy*c_dx - a_dy*b_dz*d_x + a_dy*b_x*d_dz + a_dz*b_dy*d_x - a_dz*b_x*d_dy - a_x*b_dy*d_dz + a_x*b_dz*d_dy + a_dx*b_dz*d_y - a_dx*b_y*d_dz - a_dz*b_dx*d_y + a_dz*b_y*d_dx + a_y*b_dx*d_dz - a_y*b_dz*d_dx - a_dx*b_dy*d_z + a_dx*b_z*d_dy + a_dy*b_dx*d_z - a_dy*b_z*d_dx - a_z*b_dx*d_dy + a_z*b_dy*d_dx + a_dy*c_dz*d_x - a_dy*c_x*d_dz - a_dz*c_dy*d_x + a_dz*c_x*d_dy + a_x*c_dy*d_dz - a_x*c_dz*d_dy - a_dx*c_dz*d_y + a_dx*c_y*d_dz + a_dz*c_dx*d_y - a_dz*c_y*d_dx - a_y*c_dx*d_dz + a_y*c_dz*d_dx + a_dx*c_dy*d_z - a_dx*c_z*d_dy - a_dy*c_dx*d_z + a_dy*c_z*d_dx + a_z*c_dx*d_dy - a_z*c_dy*d_dx - b_dy*c_dz*d_x + b_dy*c_x*d_dz + b_dz*c_dy*d_x - b_dz*c_x*d_dy - b_x*c_dy*d_dz + b_x*c_dz*d_dy + b_dx*c_dz*d_y - b_dx*c_y*d_dz - b_dz*c_dx*d_y + b_dz*c_y*d_dx + b_y*c_dx*d_dz - b_y*c_dz*d_dx - b_dx*c_dy*d_z + b_dx*c_z*d_dy + b_dy*c_dx*d_z - b_dy*c_z*d_dx - b_z*c_dx*d_dy + b_z*c_dy*d_dx;
  double c = a_dz*b_x*c_y - a_dz*b_y*c_x - a_x*b_dz*c_y + a_x*b_y*c_dz + a_y*b_dz*c_x - a_y*b_x*c_dz - a_dy*b_x*c_z + a_dy*b_z*c_x + a_x*b_dy*c_z - a_x*b_z*c_dy - a_z*b_dy*c_x + a_z*b_x*c_dy + a_dx*b_y*c_z - a_dx*b_z*c_y - a_y*b_dx*c_z + a_y*b_z*c_dx + a_z*b_dx*c_y - a_z*b_y*c_dx - a_dz*b_x*d_y + a_dz*b_y*d_x + a_x*b_dz*d_y - a_x*b_y*d_dz - a_y*b_dz*d_x + a_y*b_x*d_dz + a_dy*b_x*d_z - a_dy*b_z*d_x - a_x*b_dy*d_z + a_x*b_z*d_dy + a_z*b_dy*d_x - a_z*b_x*d_dy - a_dx*b_y*d_z + a_dx*b_z*d_y + a_y*b_dx*d_z - a_y*b_z*d_dx - a_z*b_dx*d_y + a_z*b_y*d_dx + a_dz*c_x*d_y - a_dz*c_y*d_x - a_x*c_dz*d_y + a_x*c_y*d_dz + a_y*c_dz*d_x - a_y*c_x*d_dz - a_dy*c_x*d_z + a_dy*c_z*d_x + a_x*c_dy*d_z - a_x*c_z*d_dy - a_z*c_dy*d_x + a_z*c_x*d_dy + a_dx*c_y*d_z - a_dx*c_z*d_y - a_y*c_dx*d_z + a_y*c_z*d_dx + a_z*c_dx*d_y - a_z*c_y*d_dx - b_dz*c_x*d_y + b_dz*c_y*d_x + b_x*c_dz*d_y - b_x*c_y*d_dz - b_y*c_dz*d_x + b_y*c_x*d_dz + b_dy*c_x*d_z - b_dy*c_z*d_x - b_x*c_dy*d_z + b_x*c_z*d_dy + b_z*c_dy*d_x - b_z*c_x*d_dy - b_dx*c_y*d_z + b_dx*c_z*d_y + b_y*c_dx*d_z - b_y*c_z*d_dx - b_z*c_dx*d_y + b_z*c_y*d_dx;
  double d = a_x*b_y*c_z - a_x*b_z*c_y - a_y*b_x*c_z + a_y*b_z*c_x + a_z*b_x*c_y - a_z*b_y*c_x - a_x*b_y*d_z + a_x*b_z*d_y + a_y*b_x*d_z - a_y*b_z*d_x - a_z*b_x*d_y + a_z*b_y*d_x + a_x*c_y*d_z - a_x*c_z*d_y - a_y*c_x*d_z + a_y*c_z*d_x + a_z*c_x*d_y - a_z*c_y*d_x - b_x*c_y*d_z + b_x*c_z*d_y + b_y*c_x*d_z - b_y*c_z*d_x - b_z*c_x*d_y + b_z*c_y*d_x;

  if (a==0) {
    return get_smallest_pos_quad_zero(b,c,d);
  }
  b/=a; c/=a; d/=a; // normalize it all
  std::vector<double> res(3);
  int real_roots_num = SolveP3(res,b,c,d);
  switch (real_roots_num) {
    case 1:
      return (res[0] >= 0) ? res[0]:INFINITY;
    case 2: {
      double max_root = max(res[0],res[1]); double min_root = min(res[0],res[1]);
      if (min_root > 0) return min_root;
      if (max_root > 0) return max_root;
      return INFINITY;
    }
    case 3: {
      std::sort(res.begin(),res.end());
      if (res[0] > 0) return res[0];
      if (res[1] > 0) return res[1];
      if (res[2] > 0) return res[2];
      return INFINITY;
    }
  }
  
}

#define TwoPi  6.28318530717958648
const double eps=1e-14;
//---------------------------------------------------------------------------
// x - array of size 3
// In case 3 real roots: => x[0], x[1], x[2], return 3
//         2 real roots: x[0], x[1],          return 2
//         1 real root : x[0], x[1] ± i*x[2], return 1
// http://math.ivanovo.ac.ru/dalgebra/Khashin/poly/index.html
int LinesearchParametrizer::SolveP3(std::vector<double>& x,double a,double b,double c) { // solve cubic equation x^3 + a*x^2 + b*x + c
  double a2 = a*a;
    double q  = (a2 - 3*b)/9; 
  double r  = (a*(2*a2-9*b) + 27*c)/54;
    double r2 = r*r;
  double q3 = q*q*q;
  double A,B;
    if(r2<q3) {
        double t=r/sqrt(q3);
    if( t<-1) t=-1;
    if( t> 1) t= 1;
        t=acos(t);
        a/=3; q=-2*sqrt(q);
        x[0]=q*cos(t/3)-a;
        x[1]=q*cos((t+TwoPi)/3)-a;
        x[2]=q*cos((t-TwoPi)/3)-a;
        return(3);
    } else {
        A =-pow(fabs(r)+sqrt(r2-q3),1./3); 
    if( r<0 ) A=-A;
    B = A==0? 0 : B=q/A;

    a/=3;
    x[0] =(A+B)-a;
        x[1] =-0.5*(A+B)-a;
        x[2] = 0.5*sqrt(3.)*(A-B);
    if(fabs(x[2])<eps) { x[2]=x[1]; return(2); }
        return(1);
    }
}