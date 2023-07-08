#ifndef MRH_ANIMAL__BEHAVIOR_INTERFACE_HPP_
#define MRH_ANIMAL__BEHAVIOR_INTERFACE_HPP_

#include "gazebo/common/Plugin.hh"
#include "gazebo/physics/physics.hh"
#include "gazebo/util/system.hh"
#include "sdf/sdf.hh"

#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <ros/subscribe_options.h>

namespace animal
{
  class BehaviorInterface
  {
  public:
    BehaviorInterface(){};
    ~BehaviorInterface(){};

    virtual void init(ros::NodeHandlePtr _ros_node_ptr,
                      sdf::ElementPtr _sdf) = 0;
    virtual bool transition() = 0;
    virtual void update(const gazebo::common::UpdateInfo &_info,
                        gazebo::physics::WorldPtr _world_ptr,
                        gazebo::physics::ActorPtr _actor_ptr) = 0;
  };
}

#endif