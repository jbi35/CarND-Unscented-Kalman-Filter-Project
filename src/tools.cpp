#include <iostream>
#include "tools.h"
#include "math.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth)
{
  // make some sanity checks
  assert(estimations.size() > 0);
  assert(estimations.size() == ground_truth.size());

  VectorXd rmse(4);
  rmse << 0.0, 0.0, 0.0, 0.0;

  //accumulate squared residuals
  for(unsigned int i=0; i < estimations.size(); ++i)
  {

  	VectorXd residual = estimations[i] - ground_truth[i];

  	//coefficient-wise multiplication
  	residual = residual.array()*residual.array();
  	rmse += residual;
  }

  //calculate the mean
  rmse = rmse/estimations.size();

  //calculate the squared root
  rmse = rmse.array().sqrt();

  //return the result
  return rmse;
}

VectorXd Tools::Polar2Cartesian(const double rho, const double phi)
{
  VectorXd result;
  result = VectorXd(2);
  result << rho * cos(phi), rho * sin(phi);
  return result;
}


void Tools::NormalizeAngle(VectorXd& z, int index)
{
  while (z(index)>M_PI)
     {
         z(index) -= 2 * M_PI;
     }
     while (z(index)<-M_PI)
     {
         z(index) += 2 * M_PI;
     }
}
