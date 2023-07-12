#include "mrh_animal/saber_flocking.hpp"

namespace animal
{
  namespace behavior
  {
    SaberFlocking::SaberFlocking(double sensing_range,
                                 double danger_range,
                                 Eigen::VectorXd initial_consensus)
        : sensing_range_(sensing_range), danger_range_(danger_range),
          consensus_(initial_consensus)
    {
    }

    SaberFlocking::~SaberFlocking()
    {
    }

    void SaberFlocking::init(ros::NodeHandlePtr _ros_node_ptr,
                             gazebo::physics::WorldPtr _world_ptr,
                             gazebo::physics::ActorPtr _actor_ptr,
                             sdf::ElementPtr _sdf)
    {
      for (unsigned int i = 0; i < _world_ptr->ModelCount(); ++i)
      {
        gazebo::physics::ModelPtr model = _world_ptr->ModelByIndex(i);
        std::string name = _actor_ptr->GetName();
        std::string model_name = model->GetName();
        if (model_name.find("herd") != std::string::npos && model_name != name)
        {
          addHerd(model_name);
        }
      }
      last_update_ = 0;
      last_herd_states_update_ = 0;
      last_control_step_ = 0;
      pre_states_ = Eigen::MatrixXd(all_herd_member_names_.size(), 4);
      pre_states_.setZero();
      pre_state_ = Eigen::VectorXd(4);
      pre_state_.setZero();
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
    void SaberFlocking::setTarget(const Eigen::VectorXd &target)
    {
      setConsensusTarget(target);
    }

    void SaberFlocking::setConsensusTarget(const Eigen::VectorXd &consensus)
    {
      consensus_ = consensus;
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
      double updatet = (_info.simTime - last_herd_states_update_).Double();
      ignition::math::Pose3d pose = _actor_ptr->WorldPose();
      ignition::math::Vector3d rpy = pose.Rot().Euler();
      Eigen::VectorXd state(4);
      state.setZero();
      if (updatet < 0.01)
      {
        // Compute the yaw orientation
        ignition::math::Vector3d pos = pose.Pos() - _actor_ptr->WorldPose().Pos();
        ignition::math::Angle yaw = atan2(pos.Y(), pos.X()) + 1.5707 - rpy.Z();
        yaw.Normalize();
        pose.Rot() = ignition::math::Quaterniond(1.5707, 0, rpy.Z() + yaw.Radian() * 0.001);
        double distance_travelled = (pose.Pos() - _actor_ptr->WorldPose().Pos())
                                        .Length();
        if (distance_travelled > 1e-6)
        {
          _actor_ptr->SetWorldPose(pose, false, false);
        }
        ignition::math::Vector3d linear;
        linear.X(state(2));
        linear.Y(state(3));
        linear.Z(0);
        ignition::math::Vector3d angular;
        _actor_ptr->SetLinearVel(linear);
        _actor_ptr->SetScriptTime(_actor_ptr->ScriptTime() +
                                  (distance_travelled * 5.1));
        return state;
      }
      state(0) = pose.Pos().X();
      state(1) = pose.Pos().Y();
      state(2) = pre_state_(2);
      state(3) = pre_state_(3);
      Eigen::MatrixXd herd_states = getAllHerdStates(_world_ptr);

      // Alpha term
      Eigen::VectorXd u_alpha = calcFlockingControl(state, herd_states);

      // Gamma term
      Eigen::VectorXd u_gamma = globalClustering(state);

      // Update state
      Eigen::VectorXd pdot = state.segment(2, 2);
      pdot = pdot + 0.075 * (0.5 * u_gamma + u_alpha);
      Eigen::VectorXd qdot = state.segment(0, 2);
      qdot = qdot + dt * pdot;
      state.segment(0, 2) = qdot;
      state.segment(2, 2) = dt * pdot;
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
      if (distance_travelled > 1e-6)
      {
        _actor_ptr->SetWorldPose(pose, false, false);
      }
      ignition::math::Vector3d linear;
      linear.X(state(2));
      linear.Y(state(3));
      linear.Z(0);
      ignition::math::Vector3d angular;
      _actor_ptr->SetLinearVel(linear);
      _actor_ptr->SetScriptTime(_actor_ptr->ScriptTime() +
                                (distance_travelled * 5.1));
      last_update_ = _info.simTime;
      last_herd_states_update_ = _info.simTime;
      pre_states_ = herd_states;
      pre_state_ = state;
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
        if (!initialised_)
        {
          herd_states(index, 2) = 0;
          herd_states(index, 3) = 0;
        }
        else
        {
          herd_states(index, 2) = herd_states(index, 0) - pre_states_(index, 0);
          herd_states(index, 3) = herd_states(index, 1) - pre_states_(index, 1);
        }
        index++;
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
      Eigen::VectorXd target = consensus_;
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
      u_alpha.setZero();

      Eigen::VectorXd qi = state.segment(0, 2);
      Eigen::VectorXd pi = state.segment(2, 2);

      // Filter
      std::vector<int> in_range_index;
      for (int i = 0; i < herd_states.rows(); i++)
      {
        Eigen::VectorXd qj_ith = herd_states.row(i);
        Eigen::VectorXd posej = qj_ith.segment(0, 2);
        Eigen::VectorXd d = qi - posej;
        if (d.norm() <= 2.5 && d.norm() > 0.2)
        {
          in_range_index.push_back(i);
        }
      }
      if (in_range_index.size() == 0)
      {
        return u_alpha;
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

      Eigen::VectorXd alpha_grad = gradientTerm(
          3.46410161514, qi, qj, 2, 2);
      Eigen::VectorXd alpha_consensus = consensusTerm(
          3.46410161514, qi, qj, pi, pj, 2);

      u_alpha = alpha_grad + alpha_consensus;
      return u_alpha;
    }
  }
}