#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

class Tools {
public:
  /**
  * Constructor.
  */
  Tools();

  /**
  * Destructor.
  */
  virtual ~Tools();

  /**
  * A helper method to calculate RMSE.
  */
  Eigen::VectorXd CalculateRMSE(const std::vector<Eigen::VectorXd> &estimations,
                                const std::vector<Eigen::VectorXd> &ground_truth);

  /**
  * A helper method to convert polar to cartesion coordinates
  */
  Eigen::VectorXd Polar2Cartesian(const double rho, const double phi);

  /**
  * A helper method to normalize angle component of state vector to -pi pi
  */
  void NormalizeAngle(Eigen::VectorXd& z, int index);

};

#endif /* TOOLS_H_ */
