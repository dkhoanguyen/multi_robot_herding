#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "mrh_animal/math_utils.hpp"

TEST(MathUtilsTest, test_sigma_1)
{
  Eigen::VectorXd z(5);
  z << 1.0, 2.0, 3.0, 4.0, 5.0;
  Eigen::VectorXd result = animal::MathUtils::sigma_1(z);
  Eigen::VectorXd cross_check(5);
  cross_check << 0.70710678, 0.89442719, 0.9486833, 0.9701425, 0.98058068;
  ASSERT_TRUE(result.isApprox(cross_check, 1e-4));
}

TEST(MathUtilsTest, test_sigma_norm)
{
  Eigen::MatrixXd z(4, 2);
  z << 1, 2,
      3, 4,
      5, 6,
      7, 8;
  Eigen::VectorXd result = animal::MathUtils::sigma_norm(z);
  Eigen::VectorXd cross_check(4);
  cross_check << 2.24744871, 8.70828693, 16.64582519, 25.07135583;
  ASSERT_TRUE(result.isApprox(cross_check, 1e-4));
}

TEST(MathUtils, test_sigma_norm_grad)
{
  Eigen::MatrixXd z(4, 2);
  z << 1, 2,
      3, 4,
      5, 6,
      7, 8;
  Eigen::MatrixXd result = animal::MathUtils::sigma_norm_grad(z);
  Eigen::MatrixXd cross_check(4, 2);
  cross_check << 0.81649658, 1.63299316,
      1.60356745, 2.13808994,
      1.87646656, 2.25175988,
      1.99593082, 2.28106379;
  ASSERT_TRUE(result.isApprox(cross_check, 1e-4));
}

TEST(MathUtils, test_bump_function)
{
  Eigen::VectorXd z(6);
  z << 0.0, 0.2, 0.4, 0.6, 0.8, 1.0;
  Eigen::VectorXd result = animal::MathUtils::bump_function(z);
  Eigen::VectorXd cross_check(6);
  cross_check << 1, 1, 0.85355339, 0.5, 0.14644661, 0;
  ASSERT_TRUE(result.isApprox(cross_check, 1e-4));
}

TEST(MathUtils, test_phi)
{
  Eigen::VectorXd z(5);
  z << 1.0, 2.0, 3.0, 4.0, 5.0;
  Eigen::VectorXd result = animal::MathUtils::phi(z);
  Eigen::VectorXd cross_check(5);
  cross_check << 3.53553391, 4.47213595, 4.74341649, 4.8507125,  4.90290338;
  ASSERT_TRUE(result.isApprox(cross_check, 1e-4));
}

TEST(MathUtils, test_phi_alpha)
{
  Eigen::VectorXd z(5);
  z << 1.0, 2.0, 3.0, 4.0, 5.0;
  Eigen::VectorXd result = animal::MathUtils::phi_alpha(z);
  Eigen::VectorXd cross_check(5); 
  cross_check << -4.99981385, -4.9998106,  -4.99980726, -4.99980383, -4.99980031;
  ASSERT_TRUE(result.isApprox(cross_check, 1e-4));
}

// Run all the tests using the gtest framework
int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}