#include "mrh_robot/herding_by_surrouding.hpp"
#include <iostream>
namespace robot
{
  namespace behavior
  {
    HerdingBySurrounding::HerdingBySurrounding(
        ros::NodeHandle nh,
        const double &Cs,
        const double &Cr,
        const double &Cv,
        const double &Co,
        const double &distance_to_target,
        const double &interagent_spacing,
        const double &obstacle_range,
        const double &sensing_range)
        : nh_(nh), cs_(Cs), cr_(Cr), cv_(Cv), co_(Co),
          distance_to_target_(distance_to_target),
          interagent_spacing_(interagent_spacing),
          obstacle_range_(obstacle_range),
          sensing_range_(sensing_range)
    {
      registerHerdStateSubs();
      odom_sub_ = nh_.subscribe<nav_msgs::Odometry>("/odom", 10, [this](const nav_msgs::OdometryConstPtr msg)
                                                    {
          std::unique_lock<std::mutex> lck(robot_odom_.mtx);
          robot_odom_.odom = *msg;
          robot_odom_.ready = true; });
      path_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/command/pose", 50);
      update_thread_timer_ = nh_.createTimer(ros::Duration(0.01), &HerdingBySurrounding::update, this);
      robot_name_ = nh.getNamespace();
      std::cout << robot_name_ << std::endl;
    }

    HerdingBySurrounding::~HerdingBySurrounding()
    {
    }

    void HerdingBySurrounding::update(const ros::TimerEvent &event)
    {
      registerHerdStateSubs();
      registerRobotStateSubs();
      // Grab herd data
      Eigen::MatrixXd herd_states(animal_odom_data_.size(), 4);
      int index = 0;
      for (auto odom_data : animal_odom_data_)
      {
        nav_msgs::Odometry odom;
        {
          std::unique_lock<std::mutex> lck(odom_data.second->mtx);
          odom = odom_data.second->odom;
        }
        Eigen::VectorXd herd_state(4);
        herd_state(0) = odom.pose.pose.position.x;
        herd_state(1) = odom.pose.pose.position.y;
        herd_state(2) = odom.twist.twist.linear.x;
        herd_state(3) = odom.twist.twist.linear.y;
        herd_states.block<1, 4>(index, 0) = herd_state;
        index++;
      }
      // Get other robot data
      Eigen::MatrixXd robot_states(robot_odom_data_.size(), 4);
      index = 0;
      for (auto odom_data : robot_odom_data_)
      {
        nav_msgs::Odometry odom;
        {
          std::unique_lock<std::mutex> lck(odom_data.second->mtx);
          odom = odom_data.second->odom;
        }
        Eigen::VectorXd robot_state(4);
        robot_state(0) = odom.pose.pose.position.x;
        robot_state(1) = odom.pose.pose.position.y;
        robot_state(2) = odom.twist.twist.linear.x;
        robot_state(3) = odom.twist.twist.linear.y;
        robot_states.block<1, 4>(index, 0) = robot_state;
        index++;
      }

      // Get robot data
      Eigen::VectorXd state(4);
      nav_msgs::Odometry odom;
      {
        std::unique_lock<std::mutex> lck(robot_odom_.mtx);
        odom = robot_odom_.odom;
      }
      state(0) = odom.pose.pose.position.x;
      state(1) = odom.pose.pose.position.y;
      state(2) = odom.twist.twist.linear.x;
      state(3) = odom.twist.twist.linear.y;

      Eigen::VectorXd qi = state.segment(0, 2);
      Eigen::VectorXd pi = state.segment(2, 2);

      // Filter
      std::vector<int> in_range_index;
      for (int i = 0; i < herd_states.rows(); i++)
      {
        Eigen::VectorXd qj_ith = herd_states.row(i);
        Eigen::VectorXd posej = qj_ith.segment(0, 2);
        Eigen::VectorXd d = qi - posej;
        if (d.norm() <= sensing_range_ && d.norm() > 0.1)
        {
          in_range_index.push_back(i);
        }
      }
      Eigen::MatrixXd filtered_states(in_range_index.size(), 4);
      int filtered_indx = 0;
      for (const int &indx : in_range_index)
      {
        filtered_states.row(filtered_indx) = herd_states.row(indx);
        filtered_indx++;
      }
      Eigen::MatrixXd qj = filtered_states.leftCols(2);
      Eigen::MatrixXd pj = filtered_states.middleCols(2, 2);

      // Edge following
      Eigen::VectorXd ps = potentialEdgeFollowing(
          cs_, distance_to_target_, qi, qj, 0.5, 0.5);

      // Robot collision avoidance
      qj = robot_states.leftCols(2);
      pj = robot_states.middleCols(2, 2);

      Eigen::VectorXd po = interRCollisionAvoidance(
          cr_, interagent_spacing_, qi, qj, 0.5, 0.5);
      
      // std::cout << po << std::endl;

      nav_msgs::Path path;
      geometry_msgs::PoseStamped target;
      target.pose.position.x = odom.pose.pose.position.x + ps(0) + po(0);
      target.pose.position.y = odom.pose.pose.position.y + ps(1) + po(1);
      target.pose.position.z = 1;
      // std::cout << ps.transpose() << std::endl;
      path_pub_.publish(target);
    }

    Eigen::VectorXd HerdingBySurrounding::multiConstAttraction(
        const double &gain, const double &d,
        const Eigen::VectorXd &qi,
        const Eigen::MatrixXd &qj,
        const bool &attract, const bool &repulse,
        const double &c, const double &m)
    {
      int a = (int)attract;
      int r = (int)repulse;
      Eigen::MatrixXd qij = qj.rowwise() - qi.transpose();
      qij = -qij;
      Eigen::VectorXd qij_norm = qij.rowwise().norm();
      Eigen::VectorXd qij_norm_d = qij_norm.array() - d;
      Eigen::VectorXd qij_norm_d_c = -qij_norm_d.array() / c;
      Eigen::VectorXd qij_norm_d_m = qij_norm_d.array() / m;
      qij_norm_d_c = qij_norm_d_c.array().exp();
      Eigen::VectorXd fx = a * Eigen::VectorXd::Ones(qij_norm.size()) - r * qij_norm_d_c;
      qij_norm_d_m = qij_norm_d_m.array().pow(2);
      Eigen::VectorXd gx = qij_norm_d_m.array().tanh();
      Eigen::VectorXd func = -fx.array() * gx.array();
      Eigen::MatrixXd unit_vec = unitVector(qij);
      unit_vec = unit_vec.array().colwise() * func.array();
      Eigen::VectorXd u = gain * unit_vec.colwise().sum();
      return u;
    }

    Eigen::VectorXd HerdingBySurrounding::potentialEdgeFollowing(
        const double &gain, const double &d,
        const Eigen::VectorXd &qi,
        const Eigen::MatrixXd &qj,
        const double &c, const double &m)
    {
      return robot::behavior::HerdingBySurrounding::multiConstAttraction(
          gain, d, qi, qj, true, true, c, m);
    }

    Eigen::VectorXd HerdingBySurrounding::interRCollisionAvoidance(
        const double &gain, const double &d,
        const Eigen::VectorXd &qi,
        const Eigen::MatrixXd &qj,
        const double &c, const double &m)
    {
      return robot::behavior::HerdingBySurrounding::multiConstAttraction(
          gain, d, qi, qj, false, true, c, m);
    }

    void HerdingBySurrounding::registerHerdStateSubs()
    {
      ros::master::V_TopicInfo master_topics;
      ros::master::getTopics(master_topics);

      for (ros::master::V_TopicInfo::iterator it = master_topics.begin(); it != master_topics.end(); it++)
      {
        const ros::master::TopicInfo &info = *it;
        std::string topic_name = info.name;
        if (topic_name.find("herd") != std::string::npos)
        {
          // std::cout << topic_name << std::endl;
          if (animal_odom_data_.find(topic_name) == animal_odom_data_.end())
          {
            std::cout << "Registering " + topic_name << std::endl;
            // Update data
            animal_odom_data_[topic_name] = std::make_shared<DataOdom>();
            animal_odom_data_[topic_name]->odom = nav_msgs::Odometry();

            // Subscriber
            animal_odom_sub_[topic_name] = nh_.subscribe<nav_msgs::Odometry>(topic_name, 10, [this, topic_name](const nav_msgs::OdometryConstPtr msg)
                                                                             {
            std::unique_lock<std::mutex> lck(animal_odom_data_[topic_name]->mtx);
            animal_odom_data_[topic_name]->odom = *msg;
            animal_odom_data_[topic_name]->ready = true; });
          }
        }
      }
    }

    void HerdingBySurrounding::registerRobotStateSubs()
    {
      ros::master::V_TopicInfo master_topics;
      ros::master::getTopics(master_topics);

      for (ros::master::V_TopicInfo::iterator it = master_topics.begin(); it != master_topics.end(); it++)
      {
        const ros::master::TopicInfo &info = *it;
        std::string topic_name = info.name;
        if (topic_name.find("firefly") != std::string::npos &&
            topic_name.find("ground_truth/odometry") != std::string::npos &&
            topic_name.find("variance") == std::string::npos)
        {
          if (robot_odom_data_.find(topic_name) == robot_odom_data_.end() &&
              topic_name.find(robot_name_) == std::string::npos)
          {
            std::cout << "Registering " + topic_name << std::endl;
            robot_odom_data_[topic_name] = std::make_shared<DataOdom>();
            robot_odom_data_[topic_name]->odom = nav_msgs::Odometry();

            robot_odom_sub_[topic_name] = nh_.subscribe<nav_msgs::Odometry>(topic_name, 10, [this, topic_name](const nav_msgs::OdometryConstPtr msg)
                                                                            {
            std::unique_lock<std::mutex> lck(robot_odom_data_[topic_name]->mtx);
            robot_odom_data_[topic_name]->odom = *msg;
            robot_odom_data_[topic_name]->ready = true; });
          }
        }
      }
    }
  }
}
int main(int argc, char **argv)
{
  ros::init(argc, argv, "robot_behavior");
  // ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
  // std::cout << "Why dont you show anything" << std::endl;
  ros::NodeHandle nh;
  robot::behavior::HerdingBySurrounding controller(
      nh, 1, 1, 1, 1, 1, 1, 1, 20);
  ros::spin();
  return 0;
}