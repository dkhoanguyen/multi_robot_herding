#ifndef MRH_ANIMAL__WANDERING_BEHAVIOR_HPP_
#define MRH_ANIMAL__WANDERING_BEHAVIOR_HPP_

#include <string>
#include <vector>

#include "gazebo/common/Plugin.hh"
#include "gazebo/physics/physics.hh"
#include "gazebo/util/system.hh"
#include "sdf/sdf.hh"
#include "behavior_interface.hpp"

namespace animal
{
  namespace behavior
  {

    class WanderingBehavior : public BehaviorInterface
    {
    public:
      WanderingBehavior();
      ~WanderingBehavior();

      void init(ros::NodeHandlePtr _ros_node_ptr,
                sdf::ElementPtr _sdf);
      bool transition();
      Eigen::VectorXd update(const gazebo::common::UpdateInfo &_info,
                  gazebo::physics::WorldPtr _world_ptr,
                  gazebo::physics::ActorPtr _actor_ptr);

    protected:
      ignition::math::Vector3d target_;
      ignition::math::Vector3d velocity_;
      double target_weight_ = 1.0;
      double obstacle_weight_ = 1.0;
      double animation_factor_ = 1.0;

      bool set_animation_ = false;
      gazebo::common::Time last_update_;
      gazebo::physics::TrajectoryInfoPtr trajectory_info_;

      void chooseNewTarget(gazebo::physics::WorldPtr world);
      void handleObstacles(ignition::math::Vector3d &pose);
    };
  }
} // namespace animal

#endif