#ifndef MRH_ANIMAL__ANIMAL_BEHAVIOR_PLUGIN_HPP_
#define MRH_ANIMAL__ANIMAL_BEHAVIOR_PLUGIN_HPP_

#include <chrono>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <unordered_map>
#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <nav_msgs/Odometry.h>

#include "gazebo/common/Plugin.hh"
#include "gazebo/physics/physics.hh"
#include "gazebo/util/system.hh"

#include "behavior_interface.hpp"
#include "wandering_behavior.hpp"

namespace gazebo
{
  typedef std::shared_ptr<animal::behavior::BehaviorInterface> BehaviorPtr;
  class GZ_PLUGIN_VISIBLE AnimalBehaviorPlugin : public ModelPlugin
  {
  public:
    AnimalBehaviorPlugin();
    ~AnimalBehaviorPlugin();

    virtual void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf);
    virtual void Reset();

  protected:
    physics::ActorPtr actor_ptr_;
    physics::WorldPtr world_ptr_;
    sdf::ElementPtr sdf_ptr_;
    std::vector<event::ConnectionPtr> connections_;

    std::vector<std::string> ignored_models_;
    physics::TrajectoryInfoPtr traj_info_;

    virtual void onUpdate(const common::UpdateInfo &_info);

  protected:
    ros::NodeHandlePtr ros_node_ptr_;
    ros::CallbackQueue ros_queue_;
    std::thread ros_queue_thread_;

    ros::Publisher odom_pub_;

    void publishOdometry(double step_time);

  protected:
    std::unordered_map<std::string, BehaviorPtr> behaviors_map_;
  };
} // namespace gazebo

#endif