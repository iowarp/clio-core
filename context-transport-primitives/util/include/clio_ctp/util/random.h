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

#ifndef CTP_SHM_INCLUDE_HSHM_SHM_UTIL_RANDOM_H_
#define CTP_SHM_INCLUDE_HSHM_SHM_UTIL_RANDOM_H_

#include <chrono>
#include <memory>
#include <random>

namespace ctp {

class Distribution {
 protected:
  std::default_random_engine generator;

 public:
  void Seed() {
    generator = std::default_random_engine(
        std::chrono::steady_clock::now().time_since_epoch().count());
  }
  void Seed(size_t seed) { generator = std::default_random_engine(seed); }
  virtual int GetInt() = 0;
  virtual double GetDouble() = 0;
  virtual size_t GetSize() = 0;
};

class CountDistribution : public Distribution {
 private:
  size_t inc_ = 1;
  size_t count_ = 0;

 public:
  void Shape(size_t inc) { inc_ = inc; }
  int GetInt() override {
    int temp = count_;
    count_ += inc_;
    return temp;
  }
  size_t GetSize() override {
    size_t temp = count_;
    count_ += inc_;
    return temp;
  };
  double GetDouble() override {
    double temp = count_;
    count_ += inc_;
    return temp;
  };
};

class NormalDistribution : public Distribution {
 private:
  std::normal_distribution<double> distribution_;

 public:
  NormalDistribution() = default;
  void Shape(double std) {
    distribution_ = std::normal_distribution<double>(0, std);
  }
  void Shape(double mean, double std) {
    distribution_ = std::normal_distribution<double>(mean, std);
  }
  int GetInt() override { return (int)round(GetDouble()); }
  size_t GetSize() override { return (size_t)round(GetDouble()); }
  double GetDouble() override { return distribution_(generator); }
};

class GammaDistribution : public Distribution {
 private:
  std::gamma_distribution<double> distribution_;

 public:
  GammaDistribution() = default;
  void Shape(double scale) {
    distribution_ = std::gamma_distribution<double>(1, scale);
  }
  void Shape(double shape, double scale) {
    distribution_ = std::gamma_distribution<double>(shape, scale);
  }
  int GetInt() override { return (int)round(GetDouble()); }
  size_t GetSize() override { return (size_t)round(GetDouble()); }
  double GetDouble() override { return distribution_(generator); }
};

class ExponentialDistribution : public Distribution {
 private:
  std::exponential_distribution<double> distribution_;

 public:
  ExponentialDistribution() = default;
  void Shape(double scale) {
    distribution_ = std::exponential_distribution<double>(scale);
  }
  int GetInt() override { return (int)round(GetDouble()); }
  size_t GetSize() override { return (size_t)round(GetDouble()); }
  double GetDouble() override { return distribution_(generator); }
};

class UniformDistribution : public Distribution {
 private:
  std::uniform_real_distribution<double> distribution_;

 public:
  UniformDistribution() = default;
  void Shape(size_t high) {
    distribution_ = std::uniform_real_distribution<double>(0, (double)high);
  }
  void Shape(double high) {
    distribution_ = std::uniform_real_distribution<double>(0, high);
  }
  void Shape(double low, double high) {
    distribution_ = std::uniform_real_distribution<double>(low, high);
  }
  int GetInt() override { return (int)round(distribution_(generator)); }
  size_t GetSize() override { return (size_t)round(distribution_(generator)); }
  double GetDouble() override { return distribution_(generator); }
};

}  // namespace ctp

#endif  // CTP_SHM_INCLUDE_HSHM_SHM_UTIL_RANDOM_H_
