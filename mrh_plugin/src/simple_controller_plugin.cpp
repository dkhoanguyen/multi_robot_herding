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

#include <functional>
#include <chrono>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <ros/subscribe_options.h>
#include <std_msgs/Float32MultiArray.h>
#include "std_msgs/Float32.h"
#include <tf/transform_broadcaster.h>

#include <ignition/math.hh>
#include "gazebo/physics/physics.hh"
#include "mrh_plugin/simple_controller_plugin.hpp"

namespace gazebo
{
  GZ_REGISTER_MODEL_PLUGIN(SimpleControllerPlugin)

#define WALKING_ANIMATION "walking"

  /////////////////////////////////////////////////
  SimpleControllerPlugin::SimpleControllerPlugin()
  {
  }

  /////////////////////////////////////////////////
  void SimpleControllerPlugin::Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
  {
    sdf = _sdf;
    actor = boost::dynamic_pointer_cast<physics::Actor>(_model);
    world = actor->GetWorld();

    connections.push_back(event::Events::ConnectWorldUpdateBegin(
        std::bind(&SimpleControllerPlugin::OnUpdate, this, std::placeholders::_1)));

    Reset();

    // Read in the target weight
    if (_sdf->HasElement("target_weight"))
      targetWeight = _sdf->Get<double>("target_weight");
    else
      targetWeight = 1.15;

    // Read in the obstacle weight
    if (_sdf->HasElement("obstacle_weight"))
      obstacleWeight = _sdf->Get<double>("obstacle_weight");
    else
      obstacleWeight = 1.5;

    // Read in the animation factor (applied in the OnUpdate function).
    if (_sdf->HasElement("animation_factor"))
      animationFactor = _sdf->Get<double>("animation_factor");
    else
      animationFactor = 4.5;

    // Add our own name to models we should ignore when avoiding obstacles.
    ignoreModels.push_back(actor->GetName());

    // Read in the other obstacles to ignore
    if (_sdf->HasElement("ignore_obstacles"))
    {
      sdf::ElementPtr modelElem =
          _sdf->GetElement("ignore_obstacles")->GetElement("model");
      while (modelElem)
      {
        ignoreModels.push_back(modelElem->Get<std::string>());
        modelElem = modelElem->GetNextElement("model");
      }
    }

    if (!ros::isInitialized())
    {
      int argc = 0;
      char **argv = NULL;
      ros::init(argc, argv, "gazebo_client",
                ros::init_options::NoSigintHandler);
    }

    rosNode.reset(new ros::NodeHandle("gazebo_client"));
    rosNode->setCallbackQueue(&rosQueue);
    rosQueueThread =
        std::thread(std::bind(&SimpleControllerPlugin::QueueThread, this));
    VelPublisher = rosNode->advertise<geometry_msgs::Twist>("/" + actor->GetName() + "/actor_vel", 1);
    ros::SubscribeOptions so =
        ros::SubscribeOptions::create<std_msgs::Float32>(
            "/vel_cmd",
            1,
            boost::bind(&SimpleControllerPlugin::OnRosMsg, this, _1),
            ros::VoidPtr(), &rosQueue);
    rosSub = rosNode->subscribe(so);
  }

  /////////////////////////////////////////////////
  void SimpleControllerPlugin::Reset()
  {
    this->velocity = 0.8;
    this->lastUpdate = 0;
    this->start_time_ = std::chrono::high_resolution_clock::now();

    if (this->sdf && this->sdf->HasElement("target"))
      this->target = this->sdf->Get<ignition::math::Vector3d>("target");
    else
      this->target = ignition::math::Vector3d(0, -5, 1.2138);

    auto skelAnims = this->actor->SkeletonAnimations();
    if (skelAnims.find(WALKING_ANIMATION) == skelAnims.end())
    {
      gzerr << "Skeleton animation " << WALKING_ANIMATION << " not found.\n";
    }
    else
    {
      // Create custom trajectory
      this->trajectoryInfoWalking.reset(new physics::TrajectoryInfo());
      this->trajectoryInfoWalking->type = WALKING_ANIMATION;
      this->trajectoryInfoWalking->duration = 1.0;

      this->actor->SetCustomTrajectory(this->trajectoryInfoWalking);
    }
  }

  /////////////////////////////////////////////////
  void SimpleControllerPlugin::ChooseNewTarget()
  {
    ignition::math::Vector3d newTarget(this->target);
    while ((newTarget - this->target).Length() < 2.0)
    {
      newTarget.X(ignition::math::Rand::DblUniform(-3, 3.5));
      newTarget.Y(ignition::math::Rand::DblUniform(-10, 2));

      for (unsigned int i = 0; i < this->world->ModelCount(); ++i)
      {
        double dist = (this->world->ModelByIndex(i)->WorldPose().Pos() - newTarget).Length();
        if (dist < 2.0)
        {
          newTarget = this->target;
          break;
        }
      }
    }
    this->target = newTarget;
  }

  /////////////////////////////////////////////////
  void SimpleControllerPlugin::HandleObstacles(ignition::math::Vector3d &_pos)
  {
    for (unsigned int i = 0; i < this->world->ModelCount(); ++i)
    {
      physics::ModelPtr model = this->world->ModelByIndex(i);
      if (std::find(this->ignoreModels.begin(), this->ignoreModels.end(),
                    model->GetName()) == this->ignoreModels.end())
      {
        ignition::math::Vector3d offset = model->WorldPose().Pos() -
                                          this->actor->WorldPose().Pos();
        double modelDist = offset.Length();
        if (modelDist < 4.0)
        {
          double invModelDist = this->obstacleWeight / modelDist;
          offset.Normalize();
          offset *= invModelDist;
          _pos -= offset;
        }
      }
    }
  }

  /////////////////////////////////////////////////
  void SimpleControllerPlugin::OnUpdate(const common::UpdateInfo &_info)
  {
    // Time delta
    double dt = (_info.simTime - this->lastUpdate).Double();

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - this->start_time_);
    if (duration.count() >= 10000)
    {
      // Create custom trajectory
      this->trajectoryInfoWalking.reset(new physics::TrajectoryInfo());
      this->trajectoryInfoWalking->type = "standing";
      this->trajectoryInfoWalking->duration = 1.0;
      this->actor->SetCustomTrajectory(this->trajectoryInfoWalking);
    }

    ignition::math::Pose3d pose = this->actor->WorldPose();
    ignition::math::Vector3d pos = this->target - pose.Pos();
    ignition::math::Vector3d rpy = pose.Rot().Euler();

    double distance = pos.Length();

    // Choose a new target position if the actor has reached its current
    // target.
    if (distance < 0.3)
    {
      this->ChooseNewTarget();
      pos = this->target - pose.Pos();
    }

    // Normalize the direction vector, and apply the target weight
    pos = pos.Normalize() * this->targetWeight;

    // Adjust the direction vector by avoiding obstacles
    this->HandleObstacles(pos);

    // Compute the yaw orientation
    ignition::math::Angle yaw = atan2(pos.Y(), pos.X()) + 1.5707 - rpy.Z();
    yaw.Normalize();

    // Rotate in place, instead of jumping.
    // if (std::abs(yaw.Radian()) > IGN_DTOR(10))
    // {
    //   pose.Rot() = ignition::math::Quaterniond(1.5707, 0, rpy.Z() + yaw.Radian() * 0.001);
    // }
    // else
    {
      pose.Pos() += pos * this->velocity * dt;
      pose.Rot() = ignition::math::Quaterniond(1.5707, 0, rpy.Z() + yaw.Radian() * 0.001);
    }

    // Make sure the actor stays within bounds
    pose.Pos().X(std::max(-3.0, std::min(3.5, pose.Pos().X())));
    pose.Pos().Y(std::max(-10.0, std::min(2.0, pose.Pos().Y())));
    pose.Pos().Z(1.2138);

    // Distance traveled is used to coordinate motion with the walking
    // animation
    double distanceTraveled = (pose.Pos() -
                               this->actor->WorldPose().Pos())
                                  .Length();

    this->actor->SetWorldPose(pose, false, false);
    this->actor->SetScriptTime(this->actor->ScriptTime() +
                               (distanceTraveled * this->animationFactor));
    this->lastUpdate = _info.simTime;
    geometry_msgs::Twist actor_vel_twist;
    VelPublisher.publish(actor_vel_twist);
  }

  /// \brief ROS helper function that processes messages
  void SimpleControllerPlugin::QueueThread()
  {
    // It gonna be really slow if you change it to 0
    static const double timeout = 0.1;
    while (this->rosNode->ok())
    {
      this->rosQueue.callAvailable(ros::WallDuration(timeout));
    }
  }

  void SimpleControllerPlugin::OnRosMsg(const std_msgs::Float32ConstPtr &_msg)
  {
    // this->SetVelocity(_msg->data);
  }
}