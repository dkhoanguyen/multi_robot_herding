#include <gtest/gtest.h>
#include <chrono>
#include <Eigen/Dense>
#include "mrh_robot/herding_by_surrouding.hpp"

class HerdingBySurroundingFixture : public ::testing::Test
{
protected:
  Eigen::VectorXd qi;
  Eigen::MatrixXd qj;
  double d = 130;
  virtual void SetUp()
  {
    qi = Eigen::VectorXd(2);
    qi << 1, 2;
    qj = Eigen::MatrixXd(3, 2);
    qj << 3, 4,
        5, 6,
        7, 8;
  }

  virtual void TearDown()
  {
  }
};

TEST_F(HerdingBySurroundingFixture, test_multiConstAttraction)
{
  double gain = 1;
  double c = 10;
  double m = 20;
  Eigen::VectorXd result = robot::behavior::HerdingBySurrounding::multiConstAttraction(
      gain, d, qi, qj, true, true, c, m);
  Eigen::VectorXd cross_check(2);
  cross_check << -444950.54236284, -444950.54236284;
  ASSERT_TRUE(result.isApprox(cross_check, 1e-4));
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}