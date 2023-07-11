#include "mrh_animal/saber_flocking.hpp"

namespace animal
{
  namespace behavior
  {
    SaberFlocking::SaberFlocking()
    {
    }

    SaberFlocking::~SaberFlocking()
    {
    }

    animal::MathUtils::Ndarray
    SaberFlocking::gradientTerm(
        const double &gain, const Eigen::VectorXd &qi,
        const Eigen::MatrixXd &qj, const double &r, const double &d)
    {
    }

    animal::MathUtils::Ndarray
    SaberFlocking::consensusTerm(
        const double &gain, const Eigen::VectorXd &qi, const Eigen::MatrixXd &qj,
        const Eigen::VectorXd &pi, const Eigen::MatrixXd &pj, const double &r)
    {
    }

    Eigen::VectorXd SaberFlocking::getAij(
        const Eigen::VectorXd &qi,
        const Eigen::MatrixXd &qj, const double &r)
    {
      Eigen::VectorXd input = r * Eigen::VectorXd::Ones(qj.rows());
      Eigen::VectorXd r_alpha = animal::MathUtils::sigma_norm(input);
      Eigen::MatrixXd qji = qj.rowwise() - qi.transpose();
      Eigen::VectorXd sigma_norm_term = animal::MathUtils::sigma_norm(qji);
      sigma_norm_term = sigma_norm_term.array() / r_alpha.array();
      Eigen::VectorXd result = animal::MathUtils::bump_function(sigma_norm_term);
      return result;
    }
    Eigen::MatrixXd SaberFlocking::getNij(
        const Eigen::VectorXd &qi,
        const Eigen::MatrixXd &qj)
    {
      Eigen::MatrixXd qji = qj.rowwise() - qi.transpose();
      Eigen::MatrixXd result = animal::MathUtils::sigma_norm_grad(qji);
      return result;
    }
  }
}