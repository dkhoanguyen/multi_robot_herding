#include "mrh_gazebo/controller.hpp"

namespace mrh_gazebo
{
  Controller::Controller(ros::NodeHandle nh)
      : nh_(nh), ld_(0.1), v_max_(0.05), v_(v_max_), w_max_(0.5),
        pos_tol_(0.005), current_waypoint_indx_(0),
        goal_reached_(true), L_(0.1), allow_reverse_(false),
        reevaluate_linear_vel_(true)
  {
    control_thread_timer_ = nh_.createTimer(ros::Duration(0.1), &Controller::controlCallback, this);
    odom_sub_ = nh_.subscribe<nav_msgs::Odometry>("/odom", 10, &Controller::odomCallback, this);
    path_sub_ = nh_.subscribe<nav_msgs::Path>("/path", 10, &Controller::pathCallback, this);
    cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
  }

  Controller::~Controller()
  {
  }

  void Controller::setPath(const std::vector<geometry_msgs::PoseStamped> &path)
  {
    std::unique_lock<std::mutex> lock(path_mtx_);
    current_waypoint_indx_ = 0;
    goal_reached_ = false;
    path_ = path;
    reevaluate_linear_vel_ = true;
  }

  void Controller::step(
      const nav_msgs::Odometry &current_odom,
      const sensor_msgs::LaserScan &scan,
      geometry_msgs::Twist &vel_cmd)
  {
    // Get lookahead for the next waypoint
    int pre_indx = current_waypoint_indx_;
    geometry_msgs::TransformStamped lookahead;

    // Extract waypoint lookahead and update next waypoint index
    std::vector<geometry_msgs::PoseStamped> path;
    {
      std::unique_lock<std::mutex> lock(path_mtx_);
      path = path_;
    }
    current_waypoint_indx_ = extractNextWaypoint(
        current_odom.pose.pose, current_waypoint_indx_, path,
        lookahead);

    bool evaluate_linear_vel_if_allow_reverse =
        (current_waypoint_indx_ != pre_indx && current_waypoint_indx_ < path.size());

    trackLookahead(current_odom.pose.pose, lookahead,
                   evaluate_linear_vel_if_allow_reverse, vel_cmd);
  }

  // For feedback
  double Controller::getDistanceToGoal(const geometry_msgs::Pose &current_pose)
  {
    std::vector<geometry_msgs::PoseStamped> path;
    {
      std::unique_lock<std::mutex> lock(path_mtx_);
      path = path_;
    }
    if (goal_reached_)
    {
      return 0;
    }

    if (path.size() == 0)
    {
      return -1;
    }

    Eigen::Vector2d current_pos(current_pose.position.x, current_pose.position.y);
    Eigen::Vector2d goal(path.back().pose.position.x, path.back().pose.position.y);
    return (goal - current_pos).norm();
  }

  // For accessing
  bool Controller::reachGoal()
  {
    return goal_reached_;
  }

  //! Compute transform that transforms a pose into the robot frame (base_link)
  Eigen::Matrix4d Controller::transformToBaseLink(
      const geometry_msgs::Pose &pose,
      const geometry_msgs::Pose &robot_tf)
  {
    Eigen::Quaterniond quaternion(pose.orientation.w,
                                  pose.orientation.x,
                                  pose.orientation.y,
                                  pose.orientation.z); // Quaternion (w, x, y, z)

    // Convert quaternion to rotation matrix
    // Pose in global (map) frame
    Eigen::Matrix3d rotation_matrix = quaternion.toRotationMatrix();
    Eigen::Matrix4d pose_transform = Eigen::Matrix4d::Identity();
    pose_transform(0, 3) = pose.position.x;
    pose_transform(1, 3) = pose.position.y;
    pose_transform(2, 3) = pose.position.z;
    pose_transform.block<3, 3>(0, 0) = rotation_matrix;

    Eigen::Quaterniond quaternion_robot(robot_tf.orientation.w,
                                        robot_tf.orientation.x,
                                        robot_tf.orientation.y,
                                        robot_tf.orientation.z); // Quaternion (w, x, y, z)

    rotation_matrix = quaternion_robot.toRotationMatrix();
    Eigen::Matrix4d base_transform = Eigen::Matrix4d::Identity();
    base_transform(0, 3) = robot_tf.position.x;
    base_transform(1, 3) = robot_tf.position.y;
    base_transform(2, 3) = robot_tf.position.z;
    base_transform.block<3, 3>(0, 0) = rotation_matrix;

    return base_transform.inverse() * pose_transform;
  }

  void Controller::trackLookahead(
      const geometry_msgs::Pose &current_pose,
      const geometry_msgs::TransformStamped &lookahead,
      const bool &evaluate_linear_if_allow_reverse,
      geometry_msgs::Twist &vel_cmd)
  {
    // We first compute the new point to track, based on our current pose,
    // path information and lookahead distance.

    // If this is a new waypoint, reevaluate linear velocity
    if (allow_reverse_)
    {
      if (evaluate_linear_if_allow_reverse)
      {
        if (reevaluate_linear_vel_)
        {
          // Get remaining path from the indx onwards
          std::vector<geometry_msgs::PoseStamped> remaining_path(path_.begin() + current_waypoint_indx_, path_.end());
          // Forward sim with positive linear vel
          double pos_dis = forwardSim(current_pose,
                                      fabs(v_max_),
                                      remaining_path);
          double neg_dis = forwardSim(current_pose,
                                      -fabs(v_max_),
                                      remaining_path);
          reevaluate_linear_vel_ = false;
        }
      }
      else
      {
        reevaluate_linear_vel_ = true;
      }
    }
    if (isApproachingFinal(path_, current_waypoint_indx_))
    {
      // We are approaching the goal,
      // This is the pose of the goal w.r.t. the base_link frame
      Eigen::Matrix4d F_bl_end = transformToBaseLink(path_.back().pose, current_pose);

      if (fabs(F_bl_end(0, 3)) <= pos_tol_)
      {
        // We have reached the goal
        goal_reached_ = true;

        // Reset the path
        path_.clear();
      }
    }

    if (!goal_reached_)
    {
      // We are tracking.
      // Compute linear velocity.
      // Right now,this is not very smart :)
      v_ = copysign(v_max_, v_);

      // Compute the angular velocity.
      // Lateral error is the y-value of the lookahead point (in base_link frame)
      double yt = lookahead.transform.translation.y;
      double ld_2 = ld_ * ld_;
      vel_cmd.angular.z = std::min(2 * v_ / ld_2 * yt, w_max_);

      // Set linear velocity for tracking.
      vel_cmd.linear.x = v_;
    }
    else
    {
      // We are at the goal!
      // Stop the vehicle
      // Stop moving.
      vel_cmd.linear.x = 0.0;
      vel_cmd.angular.z = 0.0;
    }
  }

  double Controller::forwardSim(
      const geometry_msgs::Pose &current_pose,
      const double &linear_vel,
      const std::vector<geometry_msgs::PoseStamped> &remaining_path)
  {
    // Conduct a forward simulation to the future to see whether driving forward or backward is better
    // Basically rerun the entire controller in a for loop
    double end_time = 10;
    double control_freq = 0.1;
    double remaining_distance = 0;

    double current_x = current_pose.position.x;
    double current_y = current_pose.position.y;
    double current_yaw = tf2::getYaw(current_pose.orientation);
    double linear_x = linear_vel;
    double angular_z = 0;
    double sim_time = 0;
    double distance = 0;
    bool goal_reach = false;

    int local_indx = 0;
    geometry_msgs::TransformStamped lookahead;
    while (sim_time < end_time)
    {
      // We extract the lookahead point of the next waypoint based on the current
      // pose of the robot
      local_indx = extractNextWaypoint(
          current_pose, local_indx, remaining_path,
          lookahead);

      if (local_indx >= remaining_path.size())
      {
        // We are approaching the goal,
        // which is closer than ld
        // This is the pose of the goal w.r.t. the base_link frame
        Eigen::Matrix4d F_bl_end = transformToBaseLink(remaining_path.back().pose, current_pose);

        if (fabs(F_bl_end(0, 3)) <= pos_tol_)
        {
          // We have reached the goal
          goal_reach = true;
        }
      }

      if (!goal_reach)
      {
        // Compute linear velocity.
        // Right now,this is not very smart :)
        linear_x = copysign(linear_vel, linear_x);

        // Compute the angular velocity.
        // Lateral error is the y-value of the lookahead point (in base_link frame)
        double yt = lookahead.transform.translation.y;
        double ld_2 = ld_ * ld_;
        angular_z = std::min(2 * linear_x / ld_2 * yt, w_max_);
      }
      else
      {
        // We are at the goal!
        // Stop the vehicle
        lookahead.transform = geometry_msgs::Transform();
        lookahead.transform.rotation.w = 1.0;

        // Stop moving.
        linear_x = 0.0;
        angular_z = 0.0;
      }

      Eigen::Vector3d local_vel(linear_x, 0, angular_z);
      Eigen::Vector3d vel_world = (current_pose, local_vel);

      // Apply velocities
      current_x = current_x + vel_world(0) * control_freq;
      current_y = current_y + vel_world(1) * control_freq;
      current_yaw = current_yaw + vel_world(2) * control_freq;

      if (local_indx < remaining_path.size())
      {
        double target_x = remaining_path.at(local_indx).pose.position.x;
        double target_y = remaining_path.at(local_indx).pose.position.y;
        distance = sqrt(std::pow(target_x - current_x, 2) + std::pow(target_y - current_y, 2));
      }
      else
      {
        double target_x = remaining_path.back().pose.position.x;
        double target_y = remaining_path.back().pose.position.y;
        distance = sqrt(std::pow(target_x - current_x, 2) + std::pow(target_y - current_y, 2));
      }

      if (distance <= pos_tol_)
      {
        return distance;
      }
      sim_time += control_freq;
    }
    return distance;
  }

  int Controller::extractNextWaypoint(
      const geometry_msgs::Pose &current_pose,
      const int &start_indx,
      const std::vector<geometry_msgs::PoseStamped> &path,
      geometry_msgs::TransformStamped &lookahead_tf)
  {
    int idx = start_indx;
    for (; idx < path.size(); idx++)
    {
      double x1 = path.at(idx).pose.position.x;
      double x2 = current_pose.position.x;

      double y1 = path.at(idx).pose.position.y;
      double y2 = current_pose.position.y;

      double z1 = path.at(idx).pose.position.z;
      double z2 = current_pose.position.z;

      double dist = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2) + pow(z2 - z1, 2));
      if (dist > 0.05)
      {
        // Transformed lookahead to base_link frame is lateral error
        Eigen::Matrix4d F_bl_ld = transformToBaseLink(path.at(idx).pose, current_pose);

        lookahead_tf.transform.translation.x = F_bl_ld(0, 3);
        lookahead_tf.transform.translation.y = F_bl_ld(1, 3);
        lookahead_tf.transform.translation.z = F_bl_ld(2, 3);

        Eigen::Matrix3d rotation_matrix = F_bl_ld.block<3, 3>(0, 0);
        Eigen::Quaterniond quad(rotation_matrix);
        quad.normalize();

        lookahead_tf.transform.rotation.x = quad.x();
        lookahead_tf.transform.rotation.y = quad.y();
        lookahead_tf.transform.rotation.z = quad.z();
        lookahead_tf.transform.rotation.w = quad.w();
        break;
      }
    }
    return idx;
  }

  void Controller::laserScanCallback(const sensor_msgs::LaserScanConstPtr &msg)
  {
  }
  void Controller::odomCallback(const nav_msgs::OdometryConstPtr &msg)
  {
    std::unique_lock<std::mutex> lock(odom_mtx_);
    odom_ = *msg;
  }
  void Controller::pathCallback(const nav_msgs::PathConstPtr &msg)
  {
    setPath(msg->poses);
    std::cout << "Received path" << std::endl;
  }

  void Controller::controlCallback(const ros::TimerEvent &event)
  {
    nav_msgs::Odometry current_odom;
    {
      std::unique_lock<std::mutex> lock(odom_mtx_);
      current_odom = odom_;
    }
    sensor_msgs::LaserScan laser_scan;
    geometry_msgs::Twist vel_cmd;
    step(current_odom, laser_scan, vel_cmd);
    cmd_vel_pub_.publish(vel_cmd);
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "hans_cute_controller_manager");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

  ros::NodeHandle nh;
  mrh_gazebo::Controller controller(nh);
  ros::spin();
  return 0;
}