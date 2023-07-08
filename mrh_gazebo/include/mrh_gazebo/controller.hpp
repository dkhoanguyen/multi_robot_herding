#ifndef MRH_GAZEBO__CONTROLLER_HPP_
#define MRH_GAZEBO__CONTROLLER_HPP_

#include <thread>
#include <mutex>
#include <string>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include "eigen3/Eigen/Dense"

#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/utils.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>

#include <kdl/frames.hpp>
#include <dynamic_reconfigure/server.h>

namespace mrh_gazebo
{
  class Controller
  {
  public:
    Controller(ros::NodeHandle nh);
    ~Controller();

    void setPath(const std::vector<geometry_msgs::PoseStamped> &path);

    // For feedback
    double getDistanceToGoal(const geometry_msgs::Pose &current_pose);

    // For accessing
    bool reachGoal();

  protected:
    ros::NodeHandle nh_;
    ros::Timer control_thread_timer_;
    ros::Subscriber odom_sub_;
    ros::Subscriber path_sub_;
    ros::Publisher cmd_vel_pub_;

    void laserScanCallback(const sensor_msgs::LaserScanConstPtr &msg);
    void odomCallback(const nav_msgs::OdometryConstPtr &msg);
    void pathCallback(const nav_msgs::PathConstPtr &msg);

    void controlCallback(const ros::TimerEvent& event);

  protected:
    double robot_radius_;
    double observable_range_;
    double delta_tau_;
    bool reevaluate_linear_vel_;

    int current_waypoint_indx_;
    std::mutex path_mtx_;
    std::vector<geometry_msgs::PoseStamped> path_;

    std::mutex odom_mtx_;
    nav_msgs::Odometry odom_;

    double max_linear_vel_;
    double max_angular_vel_;

    double linear_error_;
    double angular_error_;

    std::atomic<bool> at_position_;
    std::atomic<bool> reach_goal_;
    std::atomic<bool> moving_to_temp_;
    std::atomic<bool> allow_reverse_;

    // Vehicle parameters
    double L_;
    // Algorithm variables
    // Position tolerace is measured along the x-axis of the robot!
    double ld_, pos_tol_;
    // Generic control variables
    double v_max_, v_, w_max_;
    // Control variables for Ackermann steering
    // Steering angle is denoted by delta
    double delta_, delta_vel_, acc_, jerk_, delta_max_;
    int idx_;
    bool goal_reached_;

    void step(
        const nav_msgs::Odometry &current_odom,
        const sensor_msgs::LaserScan &scan,
        geometry_msgs::Twist &vel_cmd);

    void trackLookahead(
        const geometry_msgs::Pose &current_pose,
        const geometry_msgs::TransformStamped &lookahead,
        const bool &evaluate_linear_if_allow_reverse,
        geometry_msgs::Twist &vel_cmd);

    Eigen::Matrix4d transformToBaseLink(const geometry_msgs::Pose &pose,
                                   const geometry_msgs::Pose &robot_tf);

    int extractNextWaypoint(
        const geometry_msgs::Pose &current_pose,
        const int &start_indx,
        const std::vector<geometry_msgs::PoseStamped> &path,
        geometry_msgs::TransformStamped &lookahead_tf);

    double forwardSim(
        const geometry_msgs::Pose &current_pose,
        const double &linear_vel,
        const std::vector<geometry_msgs::PoseStamped> &remaining_path);

    bool isApproachingFinal(
        const std::vector<geometry_msgs::PoseStamped> &path,
        const int &indx)
    {
      return !path.empty() && indx >= path.size();
    };

    Eigen::Vector3d bodyToWorld(const geometry_msgs::Pose &current_pose,
                         const Eigen::Vector3d &local_vel)
    {
      double theta = tf2::getYaw(current_pose.orientation);
      Eigen::Matrix3d R;
      R << cos(theta), -sin(theta), 0,
          sin(theta), cos(theta), 0,
          0, 0, 1;
      Eigen::Vector3d world_vel = R * local_vel;
      return world_vel;
    };
  };
}

#endif