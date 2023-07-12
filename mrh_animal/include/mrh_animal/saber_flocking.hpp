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
      SaberFlocking(double sensing_range,
                    double danger_range,
                    Eigen::VectorXd initial_consensus);
      ~SaberFlocking();

      void init(ros::NodeHandlePtr _ros_node_ptr,
                gazebo::physics::WorldPtr _world_ptr,
                gazebo::physics::ActorPtr _actor_ptr,
                sdf::ElementPtr _sdf);
      bool transition();
      void setTarget(const Eigen::VectorXd &target);
      void addHerd(const std::string &name);
      void addShepherd(const std::string &name);
      void setConsensusTarget(const Eigen::VectorXd &consensus);
      Eigen::VectorXd update(const gazebo::common::UpdateInfo &_info,
                             gazebo::physics::WorldPtr _world_ptr,
                             gazebo::physics::ActorPtr _actor_ptr);

      static Eigen::VectorXd getAdjacencyVector(
          const Eigen::VectorXd &qi, const Eigen::MatrixXd &qj, const double &r);

      static Eigen::VectorXd gradientTerm(
          const double &gain, const Eigen::VectorXd &qi,
          const Eigen::MatrixXd &qj, const double &r, const double &d);

      static Eigen::VectorXd consensusTerm(
          const double &gain, const Eigen::VectorXd &qi, const Eigen::MatrixXd &qj,
          const Eigen::VectorXd &pi, const Eigen::MatrixXd &pj, const double &r);

      static Eigen::VectorXd groupObjectiveTerm(
          const double &gain_c1, const double &gain_c2, const Eigen::VectorXd &target,
          const Eigen::VectorXd &qi, const Eigen::VectorXd &pi);

      static Eigen::VectorXd getAij(
          const Eigen::VectorXd &qi,
          const Eigen::MatrixXd &qj, const double &r);
      static Eigen::MatrixXd getNij(
          const Eigen::VectorXd &qi,
          const Eigen::MatrixXd &qj);

    protected:
      double sensing_range_;
      double danger_range_;
      Eigen::VectorXd consensus_;

      bool set_animation_ = false;
      bool initialised_ = false;
      gazebo::common::Time last_update_;
      gazebo::common::Time last_herd_states_update_;
      gazebo::common::Time last_control_step_;
      Eigen::MatrixXd pre_states_;
      Eigen::VectorXd pre_state_;

      std::vector<std::string> all_herd_member_names_;
      std::vector<std::string> all_shepherd_names_;

      Eigen::MatrixXd getAllHerdStates(gazebo::physics::WorldPtr _world_ptr);
      Eigen::MatrixXd getHerdWithinRange(gazebo::physics::WorldPtr world_ptr,
                                         gazebo::physics::ActorPtr actor_ptr);

      Eigen::VectorXd globalClustering(const Eigen::VectorXd &state);
      Eigen::VectorXd calcFlockingControl(const Eigen::VectorXd &state, const Eigen::MatrixXd &herd_states);
    };
  }
}

#endif