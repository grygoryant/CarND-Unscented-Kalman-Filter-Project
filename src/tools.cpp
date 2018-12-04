#include <iostream>
#include "tools.h"

Tools::Tools() {}

Tools::~Tools() {}

eig_vec Tools::CalculateRMSE(const std::vector<eig_vec> &estimations,
                              const std::vector<eig_vec> &ground_truth) {
  eig_vec rmse(4);
  rmse << 0, 0, 0, 0;

  if (estimations.size() == 0 || estimations.size() != ground_truth.size())
  {
    std::cout << "Invalid estimation or ground_truth vectors" << std::endl;
    return rmse;
  }

  for (size_t i = 0; i < estimations.size(); ++i)
  {
    eig_vec residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  rmse = rmse / estimations.size();

  rmse = rmse.array().sqrt();

  return rmse;
}
