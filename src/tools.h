#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

using eig_mat = Eigen::MatrixXd;
using eig_vec = Eigen::VectorXd;

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
  eig_vec CalculateRMSE(const std::vector<eig_vec> &estimations, const std::vector<eig_vec> &ground_truth);

};

#endif /* TOOLS_H_ */