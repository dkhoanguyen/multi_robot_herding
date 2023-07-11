#include <gtest/gtest.h>
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
  double r = 40;
  virtual void SetUp()
  {
    animal::behavior::SaberFlocking flock;
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

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}