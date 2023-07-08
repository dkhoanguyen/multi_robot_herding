/*
 * Copyright (C) 2016 Open Source Robotics Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef GAZEBO_PLUGINS_SimpleControllerPlugin_HH_
#define GAZEBO_PLUGINS_SimpleControllerPlugin_HH_

#include <string>
#include <vector>

#include "gazebo/common/Plugin.hh"
#include "gazebo/physics/physics.hh"
#include "gazebo/util/system.hh"

namespace gazebo
{
  class GZ_PLUGIN_VISIBLE SimpleControllerPlugin : public ModelPlugin
  {
    /// \brief Constructor
  public:
    SimpleControllerPlugin();
    ~SimpleControllerPlugin(){};
    virtual void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf);
    virtual void Reset();

  private:
    void OnUpdate(const common::UpdateInfo &_info);
    void ChooseNewTarget();
    void HandleObstacles(ignition::math::Vector3d &_pos);

  private:
    physics::ActorPtr actor;
    physics::WorldPtr world;
    sdf::ElementPtr sdf;
    ignition::math::Vector3d velocity;
    std::vector<event::ConnectionPtr> connections;
    ignition::math::Vector3d target;

    /// \brief Target location weight (used for vector field)
  private:
    double targetWeight = 1.0;
    double obstacleWeight = 1.0;
    double animationFactor = 1.0;
    common::Time lastUpdate;
    std::chrono::_V2::system_clock::time_point start_time_;
    std::vector<std::string> ignoreModels;
    physics::TrajectoryInfoPtr trajectoryInfoWalking;

    ros::NodeHandlePtr rosNode;
    ros::Publisher VelPublisher;
    /// \brief A ROS callbackqueue that helps process messages
    ros::CallbackQueue rosQueue;

    /// \brief A thread the keeps running the rosQueue
    std::thread rosQueueThread;

  private:
    ros::Subscriber rosSub;

    void QueueThread();
    void OnRosMsg(const std_msgs::Float32ConstPtr &_msg);
  };
}
#endif