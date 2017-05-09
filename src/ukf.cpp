#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI / 2.;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  R_radar_ = MatrixXd::Zero(3,3);
  R_radar_(0,0) = pow(std_radr_,2);
  R_radar_(1,1) = pow(std_radphi_,2);
  R_radar_(2,2) = pow(std_radrd_,2);

  R_lidar_ = MatrixXd::Zero(2,2);
  R_lidar_(0,0) = pow(std_laspx_, 2);
  R_lidar_(1,1) = pow(std_laspy_, 2);

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // number of sigma points
  n_sigma_ = 2 * n_aug_ + 1;

  // object for helper functions
  tools_= Tools();

  // store process noise in vector for convenience
  process_noise_ = VectorXd::Zero(2);
  process_noise_ << pow(std_a_, 2), pow(std_yawdd_, 2);

  weights_ = ComputeWeightVector();

  is_initialized_ =false;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
  if (!is_initialized_)
  {
    Init(meas_package);
    is_initialized_ = true;
    return;
  }

  const long current_timestamp = meas_package.timestamp_;
  // get dt in seconds
  const double dt = (current_timestamp - previous_timestamp_) / 1.0e6;
  // if measurements are really close, simply skip the second one
  if (dt < 1e-3)
   return;

  Prediction(dt);
  previous_timestamp_ = meas_package.timestamp_;

  if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
  {
    UpdateLidar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
  {
    UpdateRadar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t)
{
  const MatrixXd Xsig = GenerateAugmentedSigmaPoints(x_, P_, process_noise_);
  Xsig_pred_ = PredictSigmaPoints(Xsig, delta_t);

  PredictMeanAndCovariance(&x_, &P_, Xsig_pred_);

  //tools_.NormalizeAngle(x_, 3);

  //P_ = PredictCovariance(Xsig_pred_, x_, weights_);
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  const VectorXd& z = meas_package.raw_measurements_;
  const MatrixXd Zsig = ExtractLIDARDataFromSigmaPoints(Xsig_pred_);

  // predict radar measurement and covariance
  VectorXd z_pred = VectorXd(3);
  MatrixXd S = MatrixXd(3,n_sigma_);

  PredictMeanAndCovariance(&z_pred, &S,Zsig);
  S += R_lidar_;

  const MatrixXd Tc = CalculateCrossCorrelationMatrix(Xsig_pred_, x_, Zsig, z_pred);
  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  tools_.NormalizeAngle(x_, 3);
  P_ = P_ - K*S*K.transpose();

  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package)
{
  const VectorXd& z = meas_package.raw_measurements_;
  const MatrixXd Zsig = ExtractRADARDataFromSigmaPoints(Xsig_pred_);

  // predict radar measurement and covariance
  VectorXd z_pred = VectorXd(3);
  MatrixXd S = MatrixXd(3,n_sigma_);

  PredictMeanAndCovariance(&z_pred, &S,Zsig);

  S += R_radar_;

  const MatrixXd Tc = CalculateCrossCorrelationMatrix(Xsig_pred_, x_, Zsig, z_pred);

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  tools_.NormalizeAngle(z_diff,1);

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}

void UKF::Init(const MeasurementPackage &measurement_pack)
{
  // init previous_timestamp_
  previous_timestamp_  = measurement_pack.timestamp_;

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
  {
    /**
    Convert radar from polar to cartesian coordinates and initialize state.
    */
    const double rho = measurement_pack.raw_measurements_(0,0);
    const double phi = measurement_pack.raw_measurements_(1,0);
    const double rho_dot = measurement_pack.raw_measurements_(2,0);
    VectorXd pos;
    pos = VectorXd(2);
    pos = tools_.Polar2Cartesian(rho, phi);

    x_ << pos(0,0), pos(1,0), 0.0, 0.0, 0.0;
  }
  else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER)
  {
    x_ <<  measurement_pack.raw_measurements_(0,0),
          measurement_pack.raw_measurements_(1,0),
          0.0, 0.0, 0.0;
  }
}

VectorXd UKF::ComputeWeightVector()
{

  VectorXd weights = VectorXd(2*n_aug_+1);

  // set weights
  double weight_0 = lambda_/(lambda_+n_aug_);

  weights(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; i++)
  {
    //2n+1 weights
    double weight = 0.5/(n_aug_+lambda_);
    weights(i) = weight;
  }
  return weights;
}

MatrixXd UKF::GenerateAugmentedSigmaPoints(const VectorXd& x, const MatrixXd& P,
                                           const VectorXd& process_noise)
{
  // adopted from lecture
  //create augmented state covariance
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);

  P_aug.topLeftCorner(n_x_, n_x_) = P;

  P_aug(5, 5) = process_noise(0);
  P_aug(6, 6) = process_noise(1);

  //create augmented mean state
  VectorXd x_aug = VectorXd::Zero(n_aug_);
  x_aug.head(5) = x;

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sigma_);
  const float pre_fac = sqrt(lambda_ + n_aug_);

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  Xsig_aug.col(0) = x_aug;

  for (int i = 0; i < n_aug_; ++i)
  {
    Xsig_aug.col(i + 1)          = x_aug + pre_fac * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - pre_fac * L.col(i);
  }
  return Xsig_aug;
  // TODO thing about passing matrix using pointer
}

MatrixXd UKF::PredictSigmaPoints(const MatrixXd& Xsig_aug, double delta_t)
{
  MatrixXd Xsig_pred = MatrixXd(n_x_, n_sigma_);

  //predict sigma points
  for (int i = 0; i< 2*n_aug_+1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred(0,i) = px_p;
    Xsig_pred(1,i) = py_p;
    Xsig_pred(2,i) = v_p;
    Xsig_pred(3,i) = yaw_p;
    Xsig_pred(4,i) = yawd_p;
  }

  return Xsig_pred;
}
void UKF::PredictMeanAndCovariance(VectorXd* x_out, MatrixXd* P_out,
                                   const MatrixXd& Xsig_pred)
{
  //create vector for predicted state
  VectorXd x = VectorXd::Zero(Xsig_pred.rows());
  //create covariance matrix for prediction
  MatrixXd P = MatrixXd(Xsig_pred.rows(), Xsig_pred.rows());

  //predicted state mean
  x.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {  //iterate over sigma points
    x = x+ weights_(i) * Xsig_pred.col(i);
  }

  //predicted state covariance matrix
  P.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {  //iterate over sigma points
    // state difference
    VectorXd x_diff = Xsig_pred.col(i) - x;
    //angle normalization
    //tools_.NormalizeAngle(x_diff,3)
    P = P + weights_(i) * x_diff * x_diff.transpose() ;
  }

  //write result
  *x_out = x;
  *P_out = P;
}

MatrixXd  UKF::ExtractRADARDataFromSigmaPoints( const MatrixXd& Xsig_pred)
{
  MatrixXd Zsig = MatrixXd(3,n_sigma_);
  //transform sigma points into measurement space
  for (int i = 0; i < n_sigma_ ; i++)
  {
    // extract values for better readibility
    double p_x = Xsig_pred(0,i);
    double p_y = Xsig_pred(1,i);
    double v   = Xsig_pred(2,i);
    double yaw = Xsig_pred(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }
  return Zsig;
}

MatrixXd  UKF::ExtractLIDARDataFromSigmaPoints( const MatrixXd& Xsig_pred)
{
  MatrixXd Zsig = MatrixXd(2,n_sigma_);
  for (int i = 0; i < n_sigma_ ; i++)
  {
    // extract values for better readibility
    double p_x = Xsig_pred(0,i);
    double p_y = Xsig_pred(1,i);
    // measurement model
    Zsig(0,i) = p_x; //x
    Zsig(1,i) = p_y; //y
  }
  return Zsig;
}

MatrixXd UKF::CalculateCrossCorrelationMatrix(const MatrixXd& Xsig,
                                              const VectorXd& x,
                                              const MatrixXd& Zsig,
                                              const VectorXd& z)
{
  MatrixXd Tc = MatrixXd::Zero(x.rows(), z.rows());
  for (int i = 0; i < n_sigma_; ++i)
  {
    VectorXd x_diff = Xsig.col(i) - x;
    VectorXd z_diff = Zsig.col(i) - z;
    tools_.NormalizeAngle(x_diff, 3);
    tools_.NormalizeAngle(z_diff, 1);
    Tc += weights_(i) * x_diff * z_diff.transpose();
  }
  return Tc;
}
