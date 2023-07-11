#include <gtest/gtest.h>
#include <chrono>
#include <Eigen/Dense>
#include "mrh_animal/math_utils.hpp"
#include "mrh_animal/saber_flocking.hpp"

class SaberFlockingFixture : public ::testing::Test
{
protected:
  Eigen::VectorXd qi;
  Eigen::MatrixXd qj;
  Eigen::VectorXd pi;
  Eigen::MatrixXd pj;
  Eigen::VectorXd target;
  double r = 40;
  virtual void SetUp()
  {
    qi = Eigen::VectorXd(2);
    qi << 1, 2;
    qj = Eigen::MatrixXd(3, 2);
    qj << 3, 4,
        5, 6,
        7, 8;
    pi = Eigen::VectorXd(2);
    pi << 9, 10;
    pj = Eigen::MatrixXd(3, 2);
    pj << 11, 12,
        13, 14,
        15, 16;
    target = Eigen::VectorXd(2);
    target << 20, 30;
    animal::behavior::SaberFlocking flock(40, 100, target);
  }

  virtual void TearDown()
  {
  }
};

TEST_F(SaberFlockingFixture, test_getAij)
{
  Eigen::VectorXd result = animal::behavior::SaberFlocking::getAij(qi, qj, r);
  Eigen::VectorXd cross_check(3);
  cross_check << 1, 1, 1;
  ASSERT_TRUE(result.isApprox(cross_check, 1e-4));
}

TEST_F(SaberFlockingFixture, test_getNij)
{
  Eigen::MatrixXd result = animal::behavior::SaberFlocking::getNij(qi, qj);
  Eigen::MatrixXd cross_check(3, 2);
  cross_check << 1.49071198, 1.49071198,
      1.95180015, 1.95180015,
      2.09529089, 2.09529089;
  ASSERT_TRUE(result.isApprox(cross_check, 1e-4));
}

TEST_F(SaberFlockingFixture, test_gradient_term)
{
  auto start = std::chrono::high_resolution_clock::now();
  Eigen::VectorXd result = animal::behavior::SaberFlocking::gradientTerm(
      3.46410161514, qi, qj, 40, 40);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  // Print the execution time
  std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;
  Eigen::VectorXd cross_check(2);
  cross_check << -95.91318642, -95.91318642;
  ASSERT_TRUE(result.isApprox(cross_check, 1e-4));
}

TEST_F(SaberFlockingFixture, test_consensus_term)
{
  auto start = std::chrono::high_resolution_clock::now();
  Eigen::VectorXd result = animal::behavior::SaberFlocking::consensusTerm(
      3.46410161514, qi, qj, pi, pj, 40);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  // Print the execution time
  std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;
  Eigen::VectorXd cross_check(2);
  cross_check << 41.56921938, 41.56921938;
  ASSERT_TRUE(result.isApprox(cross_check, 1e-4));
}

TEST_F(SaberFlockingFixture, test_group_objective_term)
{
  Eigen::VectorXd result = animal::behavior::SaberFlocking::groupObjectiveTerm(
      5, 0.4472135955, target, qi, pi);
  Eigen::VectorXd cross_check(2);
  cross_check << 0.96816679, 0.52467832;
  ASSERT_TRUE(result.isApprox(cross_check, 1e-4));
}

TEST_F(SaberFlockingFixture, test_adj_vector)
{
  Eigen::VectorXd result = animal::behavior::SaberFlocking::getAdjacencyVector(
      qi, qj, 40);
  Eigen::VectorXd cross_check(3);
  cross_check << 1, 1, 1;
  ASSERT_TRUE(result.isApprox(cross_check, 1e-4));
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}