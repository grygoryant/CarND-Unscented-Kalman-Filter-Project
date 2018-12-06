#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  n_x_ = 5;
  n_aug_ = n_x_ + 2;
  lambda_ = 3 - n_aug_;
  n_z_radar_ = 3;
  n_z_laser_ = 2;


  // initial state vector
  x_ = eig_vec(n_x_);

  // initial covariance matrix
  P_ = eig_mat::Identity(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.3;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
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
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

  Xsig_pred_ = eig_mat(n_x_, 2 * n_aug_ + 1);

  weights_ = eig_vec(2 * n_aug_ + 1);
  weights_(0) = lambda_/(lambda_ + n_aug_);
  for(int i = 1; i < 2 * n_aug_ + 1; ++i)
  {
      weights_(i) = 1/(2 * (lambda_ + n_aug_));
  }

  R_laser_ = eig_mat(n_z_laser_, n_z_laser_);
  R_laser_ << std_laspx_*std_laspx_, 0,
      0, std_laspy_*std_laspy_;

  R_radar_ = eig_mat(n_z_radar_, n_z_radar_);
  R_radar_ << std_radr_*std_radr_, 0, 0,
      0, std_radphi_*std_radphi_, 0,
      0, 0,std_radrd_*std_radrd_;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_)
  {
    // first measurement
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      double rho_dot = meas_package.raw_measurements_[2];
      double vx = rho_dot * cos(phi);
      double vy = rho_dot * sin(phi);
      x_ << rho * cos(phi),
          rho * sin(phi), sqrt(vx * vx + vy * vy), 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      x_ << meas_package.raw_measurements_[0],
          meas_package.raw_measurements_[1], 0, 0, 0;
    }

    time_us_ = meas_package.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0; //dt - expressed in seconds
  time_us_ = meas_package.timestamp_;
  Prediction(dt);

  if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR)
  {
    UpdateRadar(meas_package);
  }
  else if(use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER)
  {
    UpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  auto Xsig_aug = eig_mat(n_aug_, 2 * n_aug_ + 1);
  GenAugmentedSigmaPoints(Xsig_aug);
  PredictSigmaPoints(Xsig_aug, delta_t);
  PredictStateMean();
  PredictStateCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  auto z_pred = eig_vec(n_z_laser_);
  auto S = eig_mat(n_z_laser_ , n_z_laser_);
  auto Zsig = eig_mat(n_z_laser_, 2 * n_aug_ + 1);
  GenLaserMeasSigPoints(Zsig);
  PredictLaserMeasurement(Zsig, z_pred, S);

  auto Tc = eig_mat(n_x_, n_z_laser_);
  eig_vec z_diff = eig_vec::Zero(n_z_laser_);

  UpdateCommon(meas_package, z_pred, S, Zsig, Tc, z_diff);

  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
  std::cout << "NIS_L = " << NIS_laser_ << std::endl;
}

void UKF::UpdateCommon(const MeasurementPackage& meas_package,
  const eig_mat& z_pred, const eig_mat& S, const eig_mat& Zsig, 
  eig_mat& Tc, eig_vec& z_diff) {

  Tc.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    eig_vec x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  //Kalman gain K;
  eig_mat K = Tc * S.inverse();

  //residual
  z_diff = meas_package.raw_measurements_ - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  auto z_pred = eig_vec(n_z_radar_);
  auto S = eig_mat(n_z_radar_ , n_z_radar_);
  auto Zsig = eig_mat(n_z_radar_, 2 * n_aug_ + 1);
  GenRadarMeasSigPoints(Zsig);
  PredictRadarMeasurement(Zsig, z_pred, S);

  auto Tc = eig_mat(n_x_, n_z_radar_);
  eig_vec z_diff = eig_vec::Zero(n_z_radar_);

  UpdateCommon(meas_package, z_pred, S, Zsig, Tc, z_diff);

  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
  std::cout << "NIS_R = " << NIS_radar_ << std::endl;
}

void UKF::GenAugmentedSigmaPoints(eig_mat& aug_sigma_points) {
  auto x_aug = eig_vec(n_aug_);
  x_aug.setZero();
  auto P_aug = eig_mat(n_aug_, n_aug_);

  x_aug.head(n_x_) = x_;

  P_aug.fill(0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(5,5) = std_a_ * std_a_;
  P_aug(6,6) = std_yawdd_ * std_yawdd_;
  //create square root matrix
  eig_mat A = P_aug.llt().matrixL();
  //create augmented sigma points
  aug_sigma_points.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++)
  {
    aug_sigma_points.col(i + 1)     = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
    aug_sigma_points.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
  }
}

void UKF::PredictSigmaPoints(const eig_mat& aug_sigma_points, double delta_t) {
  for(int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    double p_x = aug_sigma_points(0,i);
    double p_y = aug_sigma_points(1,i);
    double v = aug_sigma_points(2,i);
    double yaw = aug_sigma_points(3,i);
    double yawd = aug_sigma_points(4,i);
    double nu_a = aug_sigma_points(5,i);
    double nu_yawdd = aug_sigma_points(6,i);
    
    double px_p, py_p;

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

    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
}

void UKF::PredictStateMean() {
  x_.setZero();
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }
}

void UKF::PredictStateCovariance() {
  P_.setZero();
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    eig_vec x_diff = Xsig_pred_.col(i) - x_;
    
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
}

void UKF::GenRadarMeasSigPoints(eig_mat& z_sig) {
  for(int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double psy = Xsig_pred_(3, i);
    double psy_dot = Xsig_pred_(4, i);
    
    double rho = sqrt(px * px + py * py) + 0.000001; // edding small value to avoid division by zero
    double phi = atan2(py, px);
    double rho_dot = (px*cos(psy)*v + py*sin(psy)*v)/rho;
    
    z_sig(0, i) = rho;
    z_sig(1, i) = phi;
    z_sig(2, i) = rho_dot;
  }
}

void UKF::GenLaserMeasSigPoints(eig_mat& z_sig) {
  for(int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);

    z_sig(0, i) = px;
    z_sig(1, i) = py;
  }
}

void UKF::PredictRadarMeasurement(const eig_mat& z_sig, eig_vec& z_pred, eig_mat& S) {
  CalcInnovCovMat(z_sig, z_pred, S);

  S = S + R_radar_;
}

void UKF::PredictLaserMeasurement(const eig_mat& z_sig, eig_vec& z_pred, eig_mat& S) {
  CalcInnovCovMat(z_sig, z_pred, S);
  
  S = S + R_laser_;
}

void UKF::CalcInnovCovMat(const eig_mat& z_sig, eig_vec& z_pred, eig_mat& S) {
  //calculate mean predicted measurement
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
      z_pred = z_pred + weights_(i) * z_sig.col(i);
  }
  
  //calculate innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  { 
    //residual
    eig_vec z_diff = z_sig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
}