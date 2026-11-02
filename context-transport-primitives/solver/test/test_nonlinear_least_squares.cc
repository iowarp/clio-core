/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "test_init.h"

using ctp::NonlinearLeastSquares;

// Test cost function: simple quadratic y = ax^2 + bx + c
void quadratic_cost_function(const std::vector<double>& params,
                            std::vector<double>& residuals,
                            const std::vector<std::pair<double, double>>& data) {
  double a = params[0];
  double b = params[1];  
  double c = params[2];
  
  residuals.resize(data.size());
  
  for (size_t i = 0; i < data.size(); ++i) {
    double x = data[i].first;
    double y_observed = data[i].second;
    double y_predicted = a * x * x + b * x + c;
    
    residuals[i] = y_observed - y_predicted;
  }
}

// Test cost function: exponential decay y = a*exp(b*x) + c
void exponential_cost_function(const std::vector<double>& params,
                              std::vector<double>& residuals,
                              const std::vector<std::pair<double, double>>& data) {
  double a = params[0];
  double b = params[1];
  double c = params[2];
  
  residuals.resize(data.size());
  
  for (size_t i = 0; i < data.size(); ++i) {
    double x = data[i].first;
    double y_observed = data[i].second;
    double y_predicted = a * std::exp(b * x) + c;
    
    residuals[i] = y_observed - y_predicted;
  }
}

TEST_CASE("NonlinearLeastSquares_QuadraticFitting") {
  // Generate synthetic data for y = 2x^2 + 3x + 1
  std::vector<std::pair<double, double>> data;
  for (double x = -2.0; x <= 2.0; x += 0.5) {
    double y = 2.0 * x * x + 3.0 * x + 1.0;
    // Add small noise
    y += 0.01 * (rand() % 100 - 50) / 100.0;
    data.push_back({x, y});
  }
  
  NonlinearLeastSquares solver;
  solver.SetParameters({1.5, 2.5, 0.5});  // Initial guess close to truth
  solver.SetTolerance(1e-10);
  solver.SetMaxIterations(100);
  
  bool converged = solver.Minimize(quadratic_cost_function, data);
  
  REQUIRE(converged);
  
  auto params = solver.GetParameters();
  REQUIRE(params.size() == 3);
  
  // Check parameters are close to expected values [2, 3, 1]
  REQUIRE(std::abs(params[0] - 2.0) < 0.1);  // a
  REQUIRE(std::abs(params[1] - 3.0) < 0.1);  // b  
  REQUIRE(std::abs(params[2] - 1.0) < 0.1);  // c
  
  // Check final cost is small
  REQUIRE(solver.GetSumOfSquares() < 0.01);
}

TEST_CASE("NonlinearLeastSquares_ExponentialFitting") {
  // Generate synthetic data for y = 4*exp(-0.5*x) + 1
  std::vector<std::pair<double, double>> data;
  for (double x = 0.0; x <= 4.0; x += 0.5) {
    double y = 4.0 * std::exp(-0.5 * x) + 1.0;
    // Add small noise
    y += 0.01 * (rand() % 100 - 50) / 100.0;
    data.push_back({x, y});
  }
  
  NonlinearLeastSquares solver;
  solver.SetParameters({3.5, -0.4, 0.8});  // Initial guess
  solver.SetTolerance(1e-8);
  solver.SetMaxIterations(200);
  
  bool converged = solver.Minimize(exponential_cost_function, data);
  
  REQUIRE(converged);
  
  auto params = solver.GetParameters();
  REQUIRE(params.size() == 3);
  
  // Check parameters are close to expected values [4, -0.5, 1]
  REQUIRE(std::abs(params[0] - 4.0) < 0.2);   // a
  REQUIRE(std::abs(params[1] - (-0.5)) < 0.1); // b
  REQUIRE(std::abs(params[2] - 1.0) < 0.2);   // c
  
  // Check final cost is reasonable
  REQUIRE(solver.GetSumOfSquares() < 0.1);
}

TEST_CASE("NonlinearLeastSquares_EmptyParameters") {
  std::vector<std::pair<double, double>> data = {{1, 2}, {2, 4}};
  
  NonlinearLeastSquares solver;
  // Don't set parameters - should fail
  
  bool converged = solver.Minimize(quadratic_cost_function, data);
  
  REQUIRE_FALSE(converged);
}

TEST_CASE("NonlinearLeastSquares_ParameterAccess") {
  NonlinearLeastSquares solver;
  
  std::vector<double> initial_params = {1.0, 2.0, 3.0};
  solver.SetParameters(initial_params);
  
  auto retrieved_params = solver.GetParameters();
  REQUIRE(retrieved_params == initial_params);
  
  // Test configuration methods
  solver.SetTolerance(1e-6);
  solver.SetMaxIterations(50);
  solver.SetLambda(0.01);
  
  // Should not crash or throw
  REQUIRE(true);
}

TEST_CASE("NonlinearLeastSquares_LinearCase") {
  // Test with linear function y = 2x + 3
  std::vector<std::pair<double, double>> data;
  for (double x = 0.0; x <= 5.0; x += 1.0) {
    double y = 2.0 * x + 3.0;
    data.push_back({x, y});
  }
  
  // Use quadratic model but with a=0
  NonlinearLeastSquares solver;
  solver.SetParameters({0.1, 1.8, 2.5});  // Close to [0, 2, 3]
  solver.SetTolerance(1e-10);
  
  bool converged = solver.Minimize(quadratic_cost_function, data);
  
  REQUIRE(converged);
  
  auto params = solver.GetParameters();
  
  // Should find a ≈ 0, b ≈ 2, c ≈ 3
  REQUIRE(std::abs(params[0]) < 0.01);      // a ≈ 0
  REQUIRE(std::abs(params[1] - 2.0) < 0.01); // b ≈ 2
  REQUIRE(std::abs(params[2] - 3.0) < 0.01); // c ≈ 3
  
  REQUIRE(solver.GetSumOfSquares() < 1e-10);
}