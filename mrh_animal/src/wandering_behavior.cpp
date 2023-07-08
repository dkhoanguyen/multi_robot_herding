#include "mrh_animal/wandering_behavior.hpp"

namespace animal
{
  WanderingBehavior::WanderingBehavior()
  {
  }

  WanderingBehavior::~WanderingBehavior()
  {
  }

  void WanderingBehavior::init(ros::NodeHandlePtr _ros_node_ptr,
                               sdf::ElementPtr _sdf)
  {
    // Read in the target weight
    if (_sdf->HasElement("target_weight"))
      target_weight_ = _sdf->Get<double>("target_weight");
    else
      target_weight_ = 1.15;

    // Read in the obstacle weight
    if (_sdf->HasElement("obstacle_weight"))
      obstacle_weight_ = _sdf->Get<double>("obstacle_weight");
    else
      obstacle_weight_ = 1.5;

    // Read in the animation factor (applied in the OnUpdate function).
    if (_sdf->HasElement("animation_factor"))
      animation_factor_ = _sdf->Get<double>("animation_factor");
    else
      animation_factor_ = 4.5;
    velocity_ = 0.8;
    last_update_ = 0;
    if (_sdf && _sdf->HasElement("target"))
      target_ = _sdf->Get<ignition::math::Vector3d>("target");
    else
      target_ = ignition::math::Vector3d(0, -5, 1.2138);
  }

  bool WanderingBehavior::transition()
  {
    return true;
  }

  void WanderingBehavior::update(const gazebo::common::UpdateInfo &_info,
                                 gazebo::physics::WorldPtr _world_ptr,
                                 gazebo::physics::ActorPtr _actor_ptr)
  {
    if (!set_animation_)
    {
      set_animation_ = true;
      gazebo::physics::TrajectoryInfoPtr traj(new gazebo::physics::TrajectoryInfo());
      traj->type = "walking";
      traj->duration = 1.0;
      _actor_ptr->SetCustomTrajectory(traj);
    }

    double dt = (_info.simTime - last_update_).Double();
    ignition::math::Pose3d pose = _actor_ptr->WorldPose();
    ignition::math::Vector3d pos = target_ - pose.Pos();
    ignition::math::Vector3d rpy = pose.Rot().Euler();

    double distance = pos.Length();

    // Choose a new target position if the actor has reached its current
    // target.
    if (distance < 0.3)
    {
      chooseNewTarget(_world_ptr);
      pos = target_ - pose.Pos();
    }

    // Normalize the direction vector, and apply the target weight
    pos = pos.Normalize() * target_weight_;

    // Adjust the direction vector by avoiding obstacles
    // handleObstacles(pos);

    // Compute the yaw orientation
    ignition::math::Angle yaw = atan2(pos.Y(), pos.X()) + 1.5707 - rpy.Z();
    yaw.Normalize();

    // // Rotate in place, instead of jumping.
    // if (std::abs(yaw.Radian()) > IGN_DTOR(10))
    // {
    //   pose.Rot() = ignition::math::Quaterniond(1.5707, 0, rpy.Z() + yaw.Radian() * 0.001);
    // }
    // else
    {
      pose.Pos() += pos * velocity_ * dt;
      pose.Rot() = ignition::math::Quaterniond(1.5707, 0, rpy.Z() + yaw.Radian() * 0.001);
    }

    // Make sure the actor stays within bounds
    pose.Pos().X(std::max(-3.0, std::min(3.5, pose.Pos().X())));
    pose.Pos().Y(std::max(-10.0, std::min(2.0, pose.Pos().Y())));
    pose.Pos().Z(1.2138);

    // Distance traveled is used to coordinate motion with the walking
    // animation
    double distance_travelled = (pose.Pos() -
                                 _actor_ptr->WorldPose().Pos())
                                    .Length();

    _actor_ptr->SetWorldPose(pose, false, false);
    _actor_ptr->SetScriptTime(_actor_ptr->ScriptTime() +
                              (distance_travelled * animation_factor_));
    last_update_ = _info.simTime;
  }

  void WanderingBehavior::chooseNewTarget(gazebo::physics::WorldPtr world)
  {
    ignition::math::Vector3d new_target(target_);
    while ((new_target - target_).Length() < 2.0)
    {
      new_target.X(ignition::math::Rand::DblUniform(-3, 3.5));
      new_target.Y(ignition::math::Rand::DblUniform(-10, 2));

      for (unsigned int i = 0; i < world->ModelCount(); ++i)
      {
        double dist = (world->ModelByIndex(i)->WorldPose().Pos() - new_target).Length();
        if (dist < 2.0)
        {
          new_target = target_;
          break;
        }
      }
    }
    target_ = new_target;
  }
  void WanderingBehavior::handleObstacles(ignition::math::Vector3d &pose)
  {
  }

}