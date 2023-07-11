#ifndef MRH_ANIMAL__MATH_UTILS_HPP_
#define MRH_ANIMAL__MATH_UTILS_HPP_

#include <Eigen/Core>
#include <cmath>

namespace animal
{
  class MathUtils
  {
  public:
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Ndarray;
    static constexpr double EPSILON = 0.1;
    static constexpr double H = 0.2;
    static constexpr double A = 5;
    static constexpr double B = 5;
    static constexpr double C = 0;

    static constexpr double R = 40;
    static constexpr double D = 40;

    static Eigen::VectorXd sigma_1(const Eigen::VectorXd &z)
    {
      Eigen::VectorXd z2 = z.array().pow(2);
      Eigen::VectorXd de = Eigen::VectorXd::Ones(z.size()) + z2;
      return z.array() / de.array().sqrt().array();
    }

    static Eigen::VectorXd sigma_norm(const Eigen::MatrixXd &z, double e = EPSILON)
    {
      Eigen::VectorXd z_row_norm = z.rowwise().norm();
      Eigen::VectorXd z_rn_2 = z_row_norm.array().pow(2);
      z_rn_2 = Eigen::VectorXd::Ones(z_rn_2.size()) + e * z_rn_2;
      Eigen::VectorXd result = z_rn_2.array().sqrt();
      result = result - Eigen::VectorXd::Ones(z_rn_2.size());
      result = result * (1 / e);
      return result;
    }

    static Eigen::MatrixXd sigma_norm_grad(const Eigen::MatrixXd &z, double e = EPSILON)
    {
      Eigen::VectorXd z_row_norm = z.rowwise().norm();
      Eigen::VectorXd z_rn_2 = z_row_norm.array().pow(2);
      z_rn_2 = Eigen::VectorXd::Ones(z_rn_2.size()) + e * z_rn_2;
      Eigen::VectorXd deno = z_rn_2.array().sqrt();
      Eigen::MatrixXd result = z.array() / deno.array().replicate(1, z.cols());
      return result;
    }

    static Eigen::VectorXd bump_function(const Eigen::VectorXd &z, double h = H)
    {
      Eigen::VectorXd ph = Eigen::VectorXd::Zero(z.size());
      for (int i = 0; i < z.size(); ++i)
      {
        if (z[i] <= 1)
          ph[i] = (1 + std::cos(M_PI * (z[i] - h) / (1 - h))) / 2;
        if (z[i] < h)
          ph[i] = 1;
        if (z[i] < 0)
          ph[i] = 0;
      }
      return ph;
    }

    static Eigen::VectorXd phi(const Eigen::VectorXd &z, double a = A, double b = B, double c = C)
    {
      return ((a + b) * sigma_1(z + Eigen::VectorXd::Ones(z.size()) * c) + (a - b) * Eigen::VectorXd::Ones(z.size())) / 2;
    }

    static Eigen::VectorXd phi_alpha(const Eigen::VectorXd &z, double r = R, double d = D)
    {
      Eigen::VectorXd result;
      Eigen::VectorXd r_alpha = sigma_norm(Eigen::VectorXd::Ones(z.size()) * r);
      Eigen::VectorXd d_alpha = sigma_norm(Eigen::VectorXd::Ones(z.size()) * d);
      Eigen::VectorXd z_r_alpha_term = z.array() / r_alpha.array();
      Eigen::VectorXd bump_term = bump_function(z_r_alpha_term);
      Eigen::VectorXd phi_term = phi(z - d_alpha);
      result = bump_term.cwiseProduct(phi_term);
      return result;
    }

    static Eigen::VectorXd normalize(const Eigen::VectorXd &v)
    {
      double n = v.norm();
      if (n < 1e-13)
        return Eigen::VectorXd::Zero(v.size());
      else
        return v / n;
    }

    // Math for bearing formation control
    static Eigen::MatrixXd orthogonal_projection_matrix(const Eigen::VectorXd &v)
    {
      int d = v.size();
      return Eigen::MatrixXd::Identity(d, d) - (v * v.transpose()) / (v.norm() * v.norm());
    }

    static Eigen::VectorXd g_ij(const Eigen::VectorXd &vi, const Eigen::VectorXd &vj)
    {
      return (vj - vi).normalized();
    }

    static double sigma(double x)
    {
      return 1 / (1 + std::exp(-x));
    }
  };

  // const double MathUtils::C = std::abs(MathUtils::A - MathUtils::B) / std::sqrt(4 * MathUtils::A * MathUtils::B);
}

#endif