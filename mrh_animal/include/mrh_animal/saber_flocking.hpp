#ifndef MRH_ANIMAL__SABER_FLOCKING_HPP_
#define MRH_ANIMAL__SABER_FLOCKING_HPP_

#include <Eigen/Core>

#include "behavior_interface.hpp"
#include "math_utils.hpp"

namespace animal
{
  namespace behavior
  {
    class SaberFlocking : public BehaviorInterface
    {
    public:
      SaberFlocking();
      ~SaberFlocking();

      void init(ros::NodeHandlePtr _ros_node_ptr,
                sdf::ElementPtr _sdf){};
      bool transition(){};
      void update(const gazebo::common::UpdateInfo &_info,
                  gazebo::physics::WorldPtr _world_ptr,
                  gazebo::physics::ActorPtr _actor_ptr){};

      static animal::MathUtils::Ndarray gradientTerm(
          const double &gain, const Eigen::VectorXd &qi,
          const Eigen::MatrixXd &qj, const double &r, const double &d);

      static animal::MathUtils::Ndarray consensusTerm(
          const double &gain, const Eigen::VectorXd &qi, const Eigen::MatrixXd &qj,
          const Eigen::VectorXd &pi, const Eigen::MatrixXd &pj, const double &r);

      static Eigen::VectorXd getAij(
          const Eigen::VectorXd &qi,
          const Eigen::MatrixXd &qj, const double &r);
      static animal::MathUtils::Ndarray getNij(
          const Eigen::VectorXd &qi,
          const Eigen::MatrixXd &qj);
    };
  }
}

#endif