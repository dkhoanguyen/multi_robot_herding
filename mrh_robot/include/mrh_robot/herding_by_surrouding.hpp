#ifndef MRH_ROBOT__HERDING_BY_SURROUNDING_HPP_
#define MRH_ROBOT__HERDING_BY_SURROUNDING_HPP_

#include <Eigen/Core>
#include <cmath>
#include <unordered_map>
#include <thread>
#include <mutex>

#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

namespace robot
{
  namespace behavior
  {
    struct DataOdom
    {
      std::mutex mtx;
      nav_msgs::Odometry odom;
      bool ready = false;
    };

    class HerdingBySurrounding
    {
    public:
      HerdingBySurrounding(
          ros::NodeHandle nh,
          const double &Cs,
          const double &Cr,
          const double &Cv,
          const double &Co,
          const double &distance_to_target,
          const double &interagent_spacing,
          const double &obstacle_range,
          const double &sensing_range);
      ~HerdingBySurrounding();

      void init(){};
      void update(const ros::TimerEvent &event);

      static Eigen::VectorXd multiConstAttraction(
          const double &gain, const double &d,
          const Eigen::VectorXd &qi,
          const Eigen::MatrixXd &qj,
          const bool &attract, const bool &repulse,
          const double &c, const double &m);

      static Eigen::VectorXd potentialEdgeFollowing(
          const double &gain, const double &d,
          const Eigen::VectorXd &qi,
          const Eigen::MatrixXd &qj,
          const double &c, const double &m);

      static Eigen::VectorXd interRCollisionAvoidance(
          const double &gain, const double &d,
          const Eigen::VectorXd &qi,
          const Eigen::MatrixXd &qj,
          const double &c, const double &m);

      static Eigen::MatrixXd unitVector(const Eigen::MatrixXd &z)
      {
        Eigen::VectorXd z_row_norm = z.rowwise().norm();
        Eigen::VectorXd deno = z_row_norm.array() + 1;
        Eigen::MatrixXd result = z.array() / deno.array().replicate(1, z.cols());
        return result;
      }

    protected:
      double cs_;
      double cr_;
      double cv_;
      double co_;
      double distance_to_target_;
      double interagent_spacing_;
      double obstacle_range_;
      double sensing_range_;

      std::unordered_map<std::string,std::shared_ptr<DataOdom>> animal_odom_data_;
      std::unordered_map<std::string,ros::Subscriber> animal_odom_sub_;

      DataOdom robot_odom_;

    protected:
      ros::NodeHandle nh_;
      ros::Timer update_thread_timer_;
      ros::Publisher path_pub_;
      ros::Subscriber odom_sub_;

      void registerHerdStateSubs();
    };
  }
}

#endif