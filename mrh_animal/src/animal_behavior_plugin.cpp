#include "mrh_animal/animal_behavior_plugin.hpp"

namespace gazebo
{
  GZ_REGISTER_MODEL_PLUGIN(AnimalBehaviorPlugin)

  AnimalBehaviorPlugin::AnimalBehaviorPlugin()
  {
  }

  AnimalBehaviorPlugin::~AnimalBehaviorPlugin()
  {
  }

  void AnimalBehaviorPlugin::Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
  {
    sdf_ptr_ = _sdf;
    actor_ptr_ = boost::dynamic_pointer_cast<physics::Actor>(_model);
    world_ptr_ = actor_ptr_->GetWorld();
    connections_.push_back(event::Events::ConnectWorldUpdateBegin(
        std::bind(&AnimalBehaviorPlugin::onUpdate, this, std::placeholders::_1)));

    // Add our own name to models we should ignore when avoiding obstacles.
    ignored_models_.push_back(actor_ptr_->GetName());

    // Read in the other obstacles to ignore
    if (sdf_ptr_->HasElement("ignore_obstacles"))
    {
      sdf::ElementPtr model_element =
          sdf_ptr_->GetElement("ignore_obstacles")->GetElement("model");
      while (model_element)
      {
        ignored_models_.push_back(model_element->Get<std::string>());
        model_element = model_element->GetNextElement("model");
      }
    }

    // Initialise ROS
    ROS_INFO("Init behavior");
    if (!ros::isInitialized())
    {
      int argc = 0;
      char **argv = NULL;
      ros::init(argc, argv, "gazebo_client",
                ros::init_options::NoSigintHandler);
    }

    ros_node_ptr_.reset(new ros::NodeHandle("gazebo_client"));
    ros_node_ptr_->setCallbackQueue(&ros_queue_);
    odom_pub_ = ros_node_ptr_->advertise<nav_msgs::Odometry>("/" + actor_ptr_->GetName() + "/odom", 1);

    BehaviorPtr wandering_behavior(new animal::behavior::WanderingBehavior());
    wandering_behavior->init(ros_node_ptr_, sdf_ptr_);
    behaviors_map_["wandering"] = wandering_behavior;
  }

  void AnimalBehaviorPlugin::Reset()
  {
  }

  void AnimalBehaviorPlugin::onUpdate(const common::UpdateInfo &_info)
  {
    // Check state transition
    std::string current_behavior_name = "";
    for (const auto &behavior : behaviors_map_)
    {
      if (behavior.second->transition())
      {
        current_behavior_name = behavior.first;
      }
    }

    BehaviorPtr current_behavior = behaviors_map_[current_behavior_name];
    current_behavior->update(_info, world_ptr_, actor_ptr_);

    // Publish odometry
    nav_msgs::Odometry current_odom;

    // Pose
    ignition::math::Pose3d pose = actor_ptr_->WorldPose();
    current_odom.pose.pose.position.x = pose.X();
    current_odom.pose.pose.position.y = pose.Y();
    current_odom.pose.pose.position.z = pose.Z();

    current_odom.pose.pose.orientation.w = pose.Rot().W();
    current_odom.pose.pose.orientation.x = pose.Rot().X();
    current_odom.pose.pose.orientation.y = pose.Rot().Y();
    current_odom.pose.pose.orientation.z = pose.Rot().Z();

    // Velocity
    ignition::math::Vector3d linear_vel = actor_ptr_->WorldLinearVel();
    ignition::math::Vector3d angular_vel = actor_ptr_->WorldAngularVel();
    current_odom.twist.twist.linear.x = linear_vel.X();
    current_odom.twist.twist.linear.y = linear_vel.Y();
    current_odom.twist.twist.linear.z = linear_vel.Z();

    current_odom.twist.twist.angular.x = angular_vel.X();
    current_odom.twist.twist.angular.y = angular_vel.Y();
    current_odom.twist.twist.angular.z = angular_vel.Z();
    odom_pub_.publish(current_odom);
  }
}