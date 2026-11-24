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

#ifndef HERMES_SHM_NONLINEAR_LEAST_SQUARES_H_
#define HERMES_SHM_NONLINEAR_LEAST_SQUARES_H_

#include <vector>
#include <cmath>
#include <algorithm>

namespace ctp {

class NonlinearLeastSquares {
 private:
  double sum_of_squares_ = 0;
  std::vector<double> parameters_;
  std::vector<double> residuals_;
  std::vector<std::vector<double>> jacobian_;
  
  // Levenberg-Marquardt parameters
  double lambda_ = 0.001;
  double lambda_factor_ = 10.0;
  double tolerance_ = 1e-8;
  int max_iterations_ = 100;
  
  // Matrix operations
  std::vector<std::vector<double>> MatrixMultiply(
      const std::vector<std::vector<double>>& A,
      const std::vector<std::vector<double>>& B) {
    size_t rows_A = A.size();
    size_t cols_A = A[0].size();
    size_t cols_B = B[0].size();
    
    std::vector<std::vector<double>> result(rows_A, std::vector<double>(cols_B, 0.0));
    
    for (size_t i = 0; i < rows_A; ++i) {
      for (size_t j = 0; j < cols_B; ++j) {
        for (size_t k = 0; k < cols_A; ++k) {
          result[i][j] += A[i][k] * B[k][j];
        }
      }
    }
    return result;
  }
  
  std::vector<std::vector<double>> MatrixTranspose(
      const std::vector<std::vector<double>>& matrix) {
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    
    std::vector<std::vector<double>> result(cols, std::vector<double>(rows));
    
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        result[j][i] = matrix[i][j];
      }
    }
    return result;
  }
  
  std::vector<std::vector<double>> MatrixInverse(
      const std::vector<std::vector<double>>& matrix) {
    size_t n = matrix.size();
    std::vector<std::vector<double>> aug(n, std::vector<double>(2 * n, 0.0));
    
    // Create augmented matrix [A|I]
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        aug[i][j] = matrix[i][j];
      }
      aug[i][i + n] = 1.0;
    }
    
    // Gauss-Jordan elimination
    for (size_t i = 0; i < n; ++i) {
      // Find pivot
      size_t pivot_row = i;
      for (size_t k = i + 1; k < n; ++k) {
        if (std::abs(aug[k][i]) > std::abs(aug[pivot_row][i])) {
          pivot_row = k;
        }
      }
      
      // Swap rows if needed
      if (pivot_row != i) {
        std::swap(aug[i], aug[pivot_row]);
      }
      
      // Make diagonal element 1
      double pivot = aug[i][i];
      if (std::abs(pivot) < 1e-12) {
        // Singular matrix, add small regularization
        aug[i][i] += 1e-12;
        pivot = aug[i][i];
      }
      
      for (size_t j = 0; j < 2 * n; ++j) {
        aug[i][j] /= pivot;
      }
      
      // Eliminate column
      for (size_t k = 0; k < n; ++k) {
        if (k != i) {
          double factor = aug[k][i];
          for (size_t j = 0; j < 2 * n; ++j) {
            aug[k][j] -= factor * aug[i][j];
          }
        }
      }
    }
    
    // Extract inverse matrix
    std::vector<std::vector<double>> inverse(n, std::vector<double>(n));
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        inverse[i][j] = aug[i][j + n];
      }
    }
    
    return inverse;
  }
  
  std::vector<double> MatrixVectorMultiply(
      const std::vector<std::vector<double>>& matrix,
      const std::vector<double>& vector) {
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    
    std::vector<double> result(rows, 0.0);
    
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        result[i] += matrix[i][j] * vector[j];
      }
    }
    return result;
  }
  
  // Numerical differentiation for Jacobian computation
  template<typename CostFunction, typename... Args>
  void ComputeJacobian(const CostFunction& cost_func, Args&&... args) {
    const double h = 1e-8;
    size_t n_params = parameters_.size();
    size_t n_residuals = residuals_.size();
    
    jacobian_.assign(n_residuals, std::vector<double>(n_params, 0.0));
    
    for (size_t j = 0; j < n_params; ++j) {
      // Forward difference
      std::vector<double> params_plus = parameters_;
      params_plus[j] += h;
      
      std::vector<double> residuals_plus;
      cost_func(params_plus, residuals_plus, std::forward<Args>(args)...);
      
      for (size_t i = 0; i < n_residuals; ++i) {
        jacobian_[i][j] = (residuals_plus[i] - residuals_[i]) / h;
      }
    }
  }
  
  double ComputeSumOfSquares() {
    double sum = 0.0;
    for (double r : residuals_) {
      sum += r * r;
    }
    return sum;
  }

 public:
  NonlinearLeastSquares() = default;
  
  void SetParameters(const std::vector<double>& initial_params) {
    parameters_ = initial_params;
  }
  
  void SetTolerance(double tol) { tolerance_ = tol; }
  void SetMaxIterations(int max_iter) { max_iterations_ = max_iter; }
  void SetLambda(double lambda) { lambda_ = lambda; }
  
  const std::vector<double>& GetParameters() const { return parameters_; }
  double GetSumOfSquares() const { return sum_of_squares_; }
  const std::vector<double>& GetResiduals() const { return residuals_; }
  
  template<typename CostFunction, typename... Args>
  bool Minimize(const CostFunction& cost_func, Args&&... args) {
    if (parameters_.empty()) {
      return false;
    }
    
    // Initial evaluation
    cost_func(parameters_, residuals_, std::forward<Args>(args)...);
    sum_of_squares_ = ComputeSumOfSquares();
    
    double prev_sum_of_squares = sum_of_squares_;
    
    for (int iter = 0; iter < max_iterations_; ++iter) {
      // Compute Jacobian
      ComputeJacobian(cost_func, std::forward<Args>(args)...);
      
      // Compute J^T * J and J^T * r
      auto J_T = MatrixTranspose(jacobian_);
      auto JTJ = MatrixMultiply(J_T, jacobian_);
      auto JTr = MatrixVectorMultiply(J_T, residuals_);
      
      // Add damping term (Levenberg-Marquardt)
      size_t n_params = parameters_.size();
      for (size_t i = 0; i < n_params; ++i) {
        JTJ[i][i] += lambda_;
      }
      
      // Solve (J^T * J + �I) * � = -J^T * r
      auto JTJ_inv = MatrixInverse(JTJ);
      auto delta = MatrixVectorMultiply(JTJ_inv, JTr);
      
      // Negate delta (we want to minimize)
      for (double& d : delta) {
        d = -d;
      }
      
      // Try new parameters
      std::vector<double> new_params = parameters_;
      for (size_t i = 0; i < n_params; ++i) {
        new_params[i] += delta[i];
      }
      
      // Evaluate cost at new parameters
      std::vector<double> new_residuals;
      cost_func(new_params, new_residuals, std::forward<Args>(args)...);
      
      double new_sum_of_squares = 0.0;
      for (double r : new_residuals) {
        new_sum_of_squares += r * r;
      }
      
      // Check if improvement
      if (new_sum_of_squares < sum_of_squares_) {
        // Accept step
        parameters_ = new_params;
        residuals_ = new_residuals;
        sum_of_squares_ = new_sum_of_squares;
        lambda_ /= lambda_factor_;  // Decrease damping
        
        // Check convergence
        if (std::abs(prev_sum_of_squares - sum_of_squares_) < tolerance_) {
          return true;
        }
        prev_sum_of_squares = sum_of_squares_;
      } else {
        // Reject step, increase damping
        lambda_ *= lambda_factor_;
      }
      
      // Prevent lambda from becoming too large
      if (lambda_ > 1e12) {
        break;
      }
    }
    
    return std::abs(prev_sum_of_squares - sum_of_squares_) < tolerance_;
  }
};

}  // namespace ctp

#endif  // HERMES_SHM_NONLINEAR_LEAST_SQUARES_H_