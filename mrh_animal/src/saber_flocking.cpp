#include "mrh_animal/saber_flocking.hpp"

namespace animal
{
  namespace behavior
  {
    SaberFlocking::SaberFlocking(double sensing_range,
                                 double danger_range,
                                 Eigen::VectorXd initial_consensus)
        : sensing_range_(sensing_range), danger_range_(danger_range),
          initial_consensus_(initial_consensus)
    {
    }

    SaberFlocking::~SaberFlocking()
    {
    }

    void SaberFlocking::init(ros::NodeHandlePtr _ros_node_ptr,
                             sdf::ElementPtr _sdf)
    {
      addHerd("herd1");
      last_update_ = 0;
    }

    bool SaberFlocking::transition()
    {
      return true;
    }

    void SaberFlocking::addHerd(const std::string &name)
    {
      all_herd_member_names_.push_back(name);
    }

    void SaberFlocking::addShepherd(const std::string &name)
    {
      all_shepherd_names_.push_back(name);
    }

    Eigen::VectorXd SaberFlocking::update(
        const gazebo::common::UpdateInfo &_info,
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
      ignition::math::Vector3d rpy = pose.Rot().Euler();
      Eigen::VectorXd state(4);
      state.setZero();
      state(0) = pose.Pos().X();
      state(1) = pose.Pos().Y();
      Eigen::MatrixXd herd_states = getHerdWithinRange(_world_ptr, _actor_ptr);
      Eigen::VectorXd u_gamma = globalClustering(state);

      Eigen::VectorXd pdot = state.segment(2, 2);
      pdot = pdot + 0.1 * u_gamma;
      Eigen::VectorXd qdot = state.segment(0, 2);
      qdot = qdot + dt * pdot;
      state.segment(0, 2) = qdot;
      state.segment(2, 2) = pdot;
      pose.Pos().X(state(0));
      pose.Pos().Y(state(1));
      pose.Pos().Z(1.2138);

      // Compute the yaw orientation
      ignition::math::Vector3d pos = pose.Pos() - _actor_ptr->WorldPose().Pos();
      ignition::math::Angle yaw = atan2(pos.Y(), pos.X()) + 1.5707 - rpy.Z();
      yaw.Normalize();
      pose.Rot() = ignition::math::Quaterniond(1.5707, 0, rpy.Z() + yaw.Radian() * 0.001);
      double distance_travelled = (pose.Pos() - _actor_ptr->WorldPose().Pos())
                                      .Length();

      _actor_ptr->SetWorldPose(pose, false, false);
      _actor_ptr->SetScriptTime(_actor_ptr->ScriptTime() +
                                (distance_travelled * 5.1));
      last_update_ = _info.simTime;
      return state;
    }

    Eigen::VectorXd
    SaberFlocking::getAdjacencyVector(
        const Eigen::VectorXd &qi, const Eigen::MatrixXd &qj, const double &r)
    {
      Eigen::VectorXd adj_vector(qj.rows());
      for (int i = 0; i < qj.rows(); i++)
      {
        Eigen::VectorXd qjth = qj.row(i);
        double range = (qjth - qi).norm();
        adj_vector(i) = (double)range <= r;
      }
      return adj_vector;
    }

    Eigen::VectorXd
    SaberFlocking::gradientTerm(
        const double &gain, const Eigen::VectorXd &qi,
        const Eigen::MatrixXd &qj, const double &r, const double &d)
    {
      Eigen::MatrixXd nij = getNij(qi, qj);
      Eigen::MatrixXd qji = qj.rowwise() - qi.transpose();
      Eigen::VectorXd sigma_norm_term = animal::MathUtils::sigma_norm(qji);
      Eigen::VectorXd phi_alpha_term = animal::MathUtils::phi_alpha(sigma_norm_term, r, d);
      Eigen::MatrixXd result = nij.array().colwise() * phi_alpha_term.array();
      Eigen::VectorXd output = gain * result.colwise().sum();
      return output;
    }

    Eigen::VectorXd
    SaberFlocking::consensusTerm(
        const double &gain, const Eigen::VectorXd &qi, const Eigen::MatrixXd &qj,
        const Eigen::VectorXd &pi, const Eigen::MatrixXd &pj, const double &r)
    {
      Eigen::VectorXd aij = getAij(qi, qj, r);
      Eigen::MatrixXd pji = pj.rowwise() - pi.transpose();
      Eigen::MatrixXd apij = pji.array().colwise() * aij.array();
      Eigen::VectorXd output = gain * apij.colwise().sum();
      return output;
    }

    Eigen::VectorXd
    SaberFlocking::groupObjectiveTerm(
        const double &gain_c1, const double &gain_c2, const Eigen::VectorXd &target,
        const Eigen::VectorXd &qi, const Eigen::VectorXd &pi)
    {
      Eigen::VectorXd pos_term = -gain_c1 * animal::MathUtils::sigma_1(qi - target);
      Eigen::VectorXd vel_term = -gain_c2 * pi;
      return pos_term + vel_term;
    }

    Eigen::VectorXd SaberFlocking::getAij(
        const Eigen::VectorXd &qi,
        const Eigen::MatrixXd &qj, const double &r)
    {
      Eigen::VectorXd input = r * Eigen::VectorXd::Ones(qj.rows());
      Eigen::VectorXd r_alpha = animal::MathUtils::sigma_norm(input);
      Eigen::MatrixXd qji = qj.rowwise() - qi.transpose();
      Eigen::VectorXd sigma_norm_term = animal::MathUtils::sigma_norm(qji);
      sigma_norm_term = sigma_norm_term.array() / r_alpha.array();
      Eigen::VectorXd result = animal::MathUtils::bump_function(sigma_norm_term);
      return result;
    }
    Eigen::MatrixXd SaberFlocking::getNij(
        const Eigen::VectorXd &qi,
        const Eigen::MatrixXd &qj)
    {
      Eigen::MatrixXd qji = qj.rowwise() - qi.transpose();
      Eigen::MatrixXd result = animal::MathUtils::sigma_norm_grad(qji);
      return result;
    }

    Eigen::MatrixXd SaberFlocking::getAllHerdStates(gazebo::physics::WorldPtr world_ptr)
    {
      Eigen::MatrixXd herd_states(all_herd_member_names_.size(), 4);
      herd_states.setZero();

      int index = 0;
      for (const std::string &herd_name : all_herd_member_names_)
      {
        gazebo::physics::ModelPtr model = world_ptr->ModelByName(herd_name);
        ignition::math::Vector3<double> pose = model->WorldPose().Pos();
        herd_states(index, 0) = pose.X();
        herd_states(index, 1) = pose.Y();
        ignition::math::Vector3<double> vel = model->WorldLinearVel();
        herd_states(index, 2) = vel.X();
        herd_states(index, 3) = vel.Y();
      }
      return herd_states;
    }
    Eigen::MatrixXd SaberFlocking::getHerdWithinRange(
        gazebo::physics::WorldPtr world_ptr,
        gazebo::physics::ActorPtr actor_ptr)
    {
      Eigen::MatrixXd herd_states(all_herd_member_names_.size(), 4);
      herd_states.setZero();

      int index = 0;
      for (const std::string &herd_name : all_herd_member_names_)
      {
        gazebo::physics::ModelPtr model = world_ptr->ModelByName(herd_name);
        ignition::math::Vector3<double> another_pose = model->WorldPose().Pos();
        ignition::math::Vector3<double> pose = actor_ptr->WorldPose().Pos();
        ignition::math::Vector3<double> d = (pose - another_pose);
        if (d.Length() <= sensing_range_)
        {
          herd_states(index, 0) = another_pose.X();
          herd_states(index, 1) = another_pose.Y();
          ignition::math::Vector3<double> vel = model->WorldLinearVel();
          herd_states(index, 2) = vel.X();
          herd_states(index, 3) = vel.Y();
        }
        index++;
      }
      return herd_states;
    }

    Eigen::VectorXd SaberFlocking::globalClustering(const Eigen::VectorXd &state)
    {
      Eigen::VectorXd u_gamma(2);
      Eigen::VectorXd target(2);
      target << 0, 0;
      Eigen::VectorXd qi = state.segment(0, 2);
      Eigen::VectorXd pi = state.segment(2, 2);
      u_gamma = animal::behavior::SaberFlocking::groupObjectiveTerm(
          5, 0.4472135955, target, qi, pi);
      return u_gamma;
    }

    Eigen::VectorXd SaberFlocking::calcFlockingControl(
        const Eigen::VectorXd &state, const Eigen::MatrixXd &herd_states)
    {
      Eigen::VectorXd u_alpha(2);
      Eigen::VectorXd qi = state.segment(0, 2);
      Eigen::VectorXd pi = state.segment(2, 2);
      Eigen::MatrixXd qj = herd_states.leftCols(2);
      Eigen::MatrixXd pj = herd_states.middleCols(2, 2);
    }
  }
}