/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**
 * Lossy Compression Parameter Study Benchmark
 *
 * This benchmark tests lossy compressors from LibPressio with floating-point data.
 * Tests how different distributions affect compression ratios for lossy algorithms.
 *
 * Tests:
 * - LibPressio ZFP compressor
 * - LibPressio bit_grooming compressor
 * - Uniform and normal distributions for float data
 */

#include "basic_test.h"
#include "hermes_shm/util/compress/compress.h"
#if HSHM_ENABLE_COMPRESS
#include "hermes_shm/util/compress/zfp.h"
#include "hermes_shm/util/compress/bitgrooming.h"
#include "hermes_shm/util/compress/fpzip.h"
#endif
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <cstring>
#include <iomanip>

#ifdef __linux__
#include <time.h>
#endif

// Benchmark result structure
struct BenchmarkResult {
    std::string library;
    std::string distribution;
    size_t chunk_size;
    double compress_time_ms;
    double decompress_time_ms;
    double compression_ratio;
    double compress_cpu_percent;
    double decompress_cpu_percent;
    double snr_db;          // Signal-to-Noise Ratio in dB
    double psnr_db;         // Peak Signal-to-Noise Ratio in dB
    double max_error;       // Maximum absolute error
    double mse;             // Mean Squared Error
    bool success;
};

// CPU usage tracking using clock_gettime with process-wide CPU measurement
struct CPUUsage {
    double cpu_time_ms;  // Total process CPU time in milliseconds (all threads)

#ifdef __linux__
    static CPUUsage getCurrent() {
        struct timespec ts;
        // CLOCK_PROCESS_CPUTIME_ID tracks CPU time for entire process (all threads)
        if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) == 0) {
            double cpu_ms = ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
            return {cpu_ms};
        }
        return {0.0};
    }
#else
    static CPUUsage getCurrent() {
        return {0.0};
    }
#endif
};

// Quality metric calculations for lossy compression
struct QualityMetrics {
    /**
     * Calculate Mean Squared Error (MSE) between original and decompressed data.
     *
     * MSE = (1/n) * Σ(original[i] - decompressed[i])²
     *
     * @param original Original float data
     * @param decompressed Decompressed float data
     * @param count Number of elements
     * @return MSE value
     */
    static double calculateMSE(const float* original, const float* decompressed, size_t count) {
        double mse = 0.0;
        for (size_t i = 0; i < count; i++) {
            double diff = static_cast<double>(original[i]) - static_cast<double>(decompressed[i]);
            mse += diff * diff;
        }
        return mse / static_cast<double>(count);
    }

    /**
     * Calculate Signal-to-Noise Ratio (SNR) in decibels.
     *
     * SNR = 10 * log10(signal_power / noise_power)
     * signal_power = Σ(original[i]²) / n
     * noise_power = MSE = Σ(original[i] - decompressed[i])² / n
     *
     * @param original Original float data
     * @param decompressed Decompressed float data
     * @param count Number of elements
     * @return SNR in dB (higher is better)
     */
    static double calculateSNR(const float* original, const float* decompressed, size_t count) {
        double signal_power = 0.0;
        double noise_power = 0.0;

        for (size_t i = 0; i < count; i++) {
            double orig = static_cast<double>(original[i]);
            double decomp = static_cast<double>(decompressed[i]);
            double diff = orig - decomp;

            signal_power += orig * orig;
            noise_power += diff * diff;
        }

        signal_power /= static_cast<double>(count);
        noise_power /= static_cast<double>(count);

        if (noise_power < 1e-10) {
            return 999.0;  // Near-perfect reconstruction (lossless)
        }

        return 10.0 * std::log10(signal_power / noise_power);
    }

    /**
     * Calculate Peak Signal-to-Noise Ratio (PSNR) in decibels.
     *
     * PSNR = 10 * log10(MAX² / MSE)
     * where MAX is the maximum possible value in the data range
     *
     * @param original Original float data
     * @param decompressed Decompressed float data
     * @param count Number of elements
     * @return PSNR in dB (higher is better)
     */
    static double calculatePSNR(const float* original, const float* decompressed, size_t count) {
        // Find maximum absolute value in original data for dynamic range
        double max_val = 0.0;
        for (size_t i = 0; i < count; i++) {
            double abs_val = std::abs(static_cast<double>(original[i]));
            if (abs_val > max_val) {
                max_val = abs_val;
            }
        }

        double mse = calculateMSE(original, decompressed, count);

        if (mse < 1e-10 || max_val < 1e-10) {
            return 999.0;  // Near-perfect reconstruction or zero signal
        }

        return 10.0 * std::log10((max_val * max_val) / mse);
    }

    /**
     * Calculate maximum absolute error between original and decompressed data.
     *
     * @param original Original float data
     * @param decompressed Decompressed float data
     * @param count Number of elements
     * @return Maximum absolute error
     */
    static double calculateMaxError(const float* original, const float* decompressed, size_t count) {
        double max_error = 0.0;
        for (size_t i = 0; i < count; i++) {
            double error = std::abs(static_cast<double>(original[i]) - static_cast<double>(decompressed[i]));
            if (error > max_error) {
                max_error = error;
            }
        }
        return max_error;
    }
};

// Data distribution generators
class DataGenerator {
public:
    // Parameterized uniform distribution for char data
    // Format: "uniform_X" where X is max value (e.g., uniform_127 = 0-127 range)
    static void generateUniformRandom(void* data, size_t size, size_t type_size, const std::string& dist_name = "uniform") {
        std::random_device rd;
        std::mt19937 gen(rd());

        // Parse max value from distribution name (e.g., "uniform_127")
        uint8_t max_val = 255;
        size_t pos = dist_name.find('_');
        if (pos != std::string::npos) {
            try {
                max_val = std::stoi(dist_name.substr(pos + 1));
            } catch (...) {
                max_val = 255;  // Default if parsing fails
            }
        }

        std::uniform_int_distribution<int> dist(0, max_val);
        uint8_t* bytes = static_cast<uint8_t*>(data);
        for (size_t i = 0; i < size * type_size; i++) {
            bytes[i] = static_cast<uint8_t>(dist(gen));
        }
    }

    // Parameterized uniform distribution for float data
    // Format: "uniform_float" - generates values in [0, 1000] range with fine precision
    static void generateUniformRandomFloat(void* data, size_t size, const std::string& dist_name = "uniform_float") {
        (void)dist_name;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1000.0f);

        float* floats = static_cast<float*>(data);
        for (size_t i = 0; i < size; i++) {
            floats[i] = dist(gen);
        }
    }

    // Parameterized normal distribution for char data
    // Format: "normal_X" where X is standard deviation (e.g., normal_10 = tight clustering)
    static void generateNormal(void* data, size_t size, size_t type_size, const std::string& dist_name = "normal") {
        std::random_device rd;
        std::mt19937 gen(rd());

        // Parse stddev from distribution name (e.g., "normal_10")
        double stddev = 30.0;
        size_t pos = dist_name.find('_');
        if (pos != std::string::npos) {
            try {
                stddev = std::stod(dist_name.substr(pos + 1));
            } catch (...) {
                stddev = 30.0;  // Default if parsing fails
            }
        }

        std::normal_distribution<> dist(128.0, stddev);
        uint8_t* bytes = static_cast<uint8_t*>(data);
        for (size_t i = 0; i < size * type_size; i++) {
            double val = dist(gen);
            // Clamp to [0, 255] range
            if (val < 0.0) {
                val = 0.0;
            }
            if (val > 255.0) {
                val = 255.0;
            }
            bytes[i] = static_cast<uint8_t>(val);
        }
    }

    // Parameterized normal distribution for float data
    // Format: "normal_float" - generates values with mean=500, stddev=200
    static void generateNormalFloat(void* data, size_t size, const std::string& dist_name = "normal_float") {
        (void)dist_name;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(500.0F, 200.0F);

        float* floats = static_cast<float*>(data);
        for (size_t i = 0; i < size; i++) {
            floats[i] = dist(gen);
        }
    }

    // HIGHLY COMPRESSIBLE: Smooth repeating pattern (sine wave)
    // Expected ratio: ~50-200x with lossy compressors
    static void generateRepeatingFloat(void* data, size_t size, const std::string& dist_name = "repeating_float") {
        (void)dist_name;
        float* floats = static_cast<float*>(data);
        for (size_t i = 0; i < size; i++) {
            // Simple repeating sine wave pattern
            float x = static_cast<float>(i % 100);  // Repeat every 100 samples
            floats[i] = std::sin(x * 0.0628F) * 1000.0F;  // 0.0628 ≈ 2π/100
        }
    }

    // MEDIUM COMPRESSIBLE: Structured data with some variation
    // Expected ratio: ~5-20x with lossy compressors
    static void generateStructuredFloat(void* data, size_t size, const std::string& dist_name = "structured_float") {
        (void)dist_name;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> noise_dist(0.0F, 10.0F);  // Small noise

        float* floats = static_cast<float*>(data);
        for (size_t i = 0; i < size; i++) {
            // Slowly varying smooth signal + small noise
            float x = static_cast<float>(i);
            float signal = std::sin(x * 0.01F) * 500.0F +
                          std::cos(x * 0.003F) * 300.0F;
            float noise = noise_dist(gen);
            floats[i] = signal + noise;
        }
    }

    // LIGHTLY COMPRESSIBLE: Noisy signal (signal + significant noise)
    // Expected ratio: ~1.5-5x with lossy compressors
    static void generateNoisyFloat(void* data, size_t size, const std::string& dist_name = "noisy_float") {
        (void)dist_name;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> noise_dist(0.0F, 200.0F);  // Large noise

        float* floats = static_cast<float*>(data);
        for (size_t i = 0; i < size; i++) {
            // Weak signal buried in noise
            float x = static_cast<float>(i);
            float signal = std::sin(x * 0.01F) * 100.0F;  // Weak signal
            float noise = noise_dist(gen);                // Strong noise
            floats[i] = signal + noise;
        }
    }

    // INCOMPRESSIBLE: Pure random noise (white noise)
    // Expected ratio: ~1.0-1.2x with lossy compressors
    static void generateRandomFloat(void* data, size_t size, const std::string& dist_name = "random_float") {
        (void)dist_name;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1000.0F, 1000.0F);

        float* floats = static_cast<float*>(data);
        for (size_t i = 0; i < size; i++) {
            floats[i] = dist(gen);  // Pure random noise
        }
    }

    static void generateGamma(void* data, size_t size, size_t type_size, const std::string& dist_name = "gamma") {
        std::random_device rd;
        std::mt19937 gen(rd());

        // Parse parameters from distribution name
        // Format: "gamma_incomp", "gamma_light", "gamma_medium", "gamma_high"
        double shape = 2.0;   // Default shape (α)
        double scale = 2.0;   // Default scale (β)
        double multiplier = 20.0;  // Scaling factor

        size_t pos = dist_name.find('_');
        if (pos != std::string::npos) {
            std::string param = dist_name.substr(pos + 1);

            if (param == "incomp") {
                // INCOMPRESSIBLE (~1.0x): Wide spread with noise to destroy patterns
                // Gamma(5, 5) has mean=25, scale by 5 → mean~125
                // Add uniform noise to spread values across full 0-255 range
                shape = 5.0;
                scale = 5.0;
                multiplier = 5.0;

                std::gamma_distribution<> gamma_dist(shape, scale);
                std::uniform_int_distribution<int> noise_dist(-30, 30);

                for (size_t i = 0; i < size * type_size; i++) {
                    double gamma_val = gamma_dist(gen) * multiplier;
                    int noise = noise_dist(gen);
                    double val = gamma_val + noise;
                    if (val < 0.0) val = 0.0;
                    if (val > 255.0) val = 255.0;
                    static_cast<uint8_t*>(data)[i] = static_cast<uint8_t>(val);
                }
                return;
            }
            if (param == "light") {
                // LIGHTLY COMPRESSIBLE (~1.1x): Wide spread, limited clustering
                // Gamma(5, 8) has mean=40, scale by 4 → mean~160, wide spread
                shape = 5.0;
                scale = 8.0;
                multiplier = 4.0;
            }
            if (param == "medium") {
                // MEDIUM COMPRESSIBLE (~1.75x): Moderate clustering
                // Gamma(2, 4) has mean=8, scale by 15 → mean~120, moderate clustering
                shape = 2.0;
                scale = 4.0;
                multiplier = 15.0;
            }
            if (param == "high") {
                // HIGHLY COMPRESSIBLE (>3x): Tight clustering at low values
                // Gamma(1, 2) has mean=2, scale by 20 → mean~40, very tight clustering
                shape = 1.0;
                scale = 2.0;
                multiplier = 20.0;
            }
        }

        std::gamma_distribution<> dist(shape, scale);
        uint8_t* bytes = static_cast<uint8_t*>(data);
        for (size_t i = 0; i < size * type_size; i++) {
            double val = dist(gen) * multiplier;
            if (val > 255.0) {
                val = 255.0;
            }
            bytes[i] = static_cast<uint8_t>(val);
        }
    }

    static void generateExponential(void* data, size_t size, size_t type_size, const std::string& dist_name = "exponential") {
        std::random_device rd;
        std::mt19937 gen(rd());

        // Parse parameters from distribution name
        // Format: "exponential_incomp", "exponential_light", "exponential_medium", "exponential_high"
        //
        // Exponential distribution: mean = 1/λ (rate)
        // Larger λ = faster decay = tighter clustering near zero
        // Smaller λ = slower decay = wider spread
        double rate = 0.05;  // Default rate (λ)
        double offset = 0.0;  // Offset to shift distribution
        double scale = 1.0;   // Scaling factor

        size_t pos = dist_name.find('_');
        if (pos != std::string::npos) {
            std::string param = dist_name.substr(pos + 1);

            if (param == "incomp") {
                // INCOMPRESSIBLE (~1.0x): Wide spread with noise to destroy patterns
                // Exponential(0.01) has mean=100, scale by 1.5 → mean~150
                // Add uniform noise to spread values across full range
                rate = 0.01;    // Slow decay (mean = 100)
                scale = 1.5;
                offset = 0.0;

                std::exponential_distribution<> exp_dist(rate);
                std::uniform_int_distribution<int> noise_dist(-50, 50);

                for (size_t i = 0; i < size * type_size; i++) {
                    double exp_val = exp_dist(gen) * scale;
                    int noise = noise_dist(gen);
                    double val = exp_val + noise;
                    if (val < 0.0) val = 0.0;
                    if (val > 255.0) val = 255.0;
                    static_cast<uint8_t*>(data)[i] = static_cast<uint8_t>(val);
                }
                return;
            }
            if (param == "light") {
                // LIGHTLY COMPRESSIBLE (~1.1x): Slow decay, wide spread
                // Exponential(0.012) has mean=83.3, scale by 2.5, offset +10 → mean~218
                rate = 0.012;   // Slow decay (mean = 83.3)
                scale = 2.5;
                offset = 10.0;
            }
            if (param == "medium") {
                // MEDIUM COMPRESSIBLE (~1.75x): Moderate decay, some clustering
                // Exponential(0.03) has mean=33.3, scale by 6 → mean~200
                rate = 0.03;    // Moderate decay (mean = 33.3)
                scale = 6.0;
                offset = 0.0;
            }
            if (param == "high") {
                // HIGHLY COMPRESSIBLE (>3x): Fast decay, tight clustering near zero
                // Exponential(0.08) has mean=12.5, scale by 8 → mean~100, tight clustering
                rate = 0.08;    // Fast decay (mean = 12.5)
                scale = 8.0;
                offset = 0.0;
            }
        }

        std::exponential_distribution<> dist(rate);
        uint8_t* bytes = static_cast<uint8_t*>(data);
        for (size_t i = 0; i < size * type_size; i++) {
            double val = dist(gen) * scale + offset;
            if (val < 0.0) val = 0.0;
            if (val > 255.0) val = 255.0;
            bytes[i] = static_cast<uint8_t>(val);
        }
    }

    static void generateRepeating(void* data, size_t size, size_t type_size, const std::string& dist_name = "repeating") {
        (void)dist_name;
        uint8_t* bytes = static_cast<uint8_t*>(data);
        for (size_t i = 0; i < size * type_size; i++) {
            bytes[i] = static_cast<uint8_t>((i / 16) % 256);
        }
    }
};

// Run benchmark for a single configuration
BenchmarkResult benchmarkCompressor(hshm::Compressor* compressor,
                                     const char* lib_name,
                                     const std::string& distribution,
                                     size_t chunk_size) {
    BenchmarkResult result;
    result.library = lib_name;
    result.distribution = distribution;
    result.chunk_size = chunk_size;
    result.success = false;

    // Generate input data (char data type only)
    std::vector<uint8_t> input_data(chunk_size);

    if (distribution.find("uniform") == 0) {
        DataGenerator::generateUniformRandom(input_data.data(), chunk_size, 1, distribution);
    } else if (distribution.find("normal") == 0) {
        DataGenerator::generateNormal(input_data.data(), chunk_size, 1, distribution);
    } else if (distribution.find("gamma") == 0) {
        DataGenerator::generateGamma(input_data.data(), chunk_size, 1, distribution);
    } else if (distribution.find("exponential") == 0) {
        DataGenerator::generateExponential(input_data.data(), chunk_size, 1, distribution);
    } else if (distribution == "repeating") {
        DataGenerator::generateRepeating(input_data.data(), chunk_size, 1, distribution);
    }

    // Allocate output buffers
    std::vector<uint8_t> compressed_data(chunk_size * 2);  // Oversized
    std::vector<uint8_t> decompressed_data(chunk_size);

    // Measure compression
    size_t cmpr_size = compressed_data.size();
    CPUUsage cpu_before = CPUUsage::getCurrent();
    auto start = std::chrono::high_resolution_clock::now();

    bool comp_ok = compressor->Compress(compressed_data.data(), cmpr_size,
                                        input_data.data(), chunk_size);

    auto end = std::chrono::high_resolution_clock::now();
    CPUUsage cpu_after = CPUUsage::getCurrent();

    if (!comp_ok || cmpr_size == 0) {
        return result;
    }

    auto compress_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    result.compress_time_ms = compress_duration.count() / 1000000.0;

    // Calculate CPU utilization percentage
    if (result.compress_time_ms > 0.0) {
        double cpu_time_ms = cpu_after.cpu_time_ms - cpu_before.cpu_time_ms;
        result.compress_cpu_percent = (cpu_time_ms / result.compress_time_ms) * 100.0;
    } else {
        result.compress_cpu_percent = 0.0;
    }

    // Compression ratio
    if (cmpr_size > 0) {
        result.compression_ratio = static_cast<double>(chunk_size) / cmpr_size;
    } else {
        result.compression_ratio = 0.0;
    }

    // Measure decompression
    size_t decmpr_size = chunk_size;
    cpu_before = CPUUsage::getCurrent();
    start = std::chrono::high_resolution_clock::now();

    bool decomp_ok = compressor->Decompress(decompressed_data.data(), decmpr_size,
                                            compressed_data.data(), cmpr_size);

    end = std::chrono::high_resolution_clock::now();
    cpu_after = CPUUsage::getCurrent();

    if (!decomp_ok || decmpr_size != chunk_size) {
        return result;
    }

    auto decompress_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    result.decompress_time_ms = decompress_duration.count() / 1000000.0;

    // Calculate CPU utilization percentage
    if (result.decompress_time_ms > 0.0) {
        double cpu_time_ms = cpu_after.cpu_time_ms - cpu_before.cpu_time_ms;
        result.decompress_cpu_percent = (cpu_time_ms / result.decompress_time_ms) * 100.0;
    } else {
        result.decompress_cpu_percent = 0.0;
    }

    // Verify correctness
    if (std::memcmp(input_data.data(), decompressed_data.data(), chunk_size) != 0) {
        return result;
    }

    result.success = true;
    return result;
}

// Run benchmark for a single configuration with floating-point data
// Used for lossy compressors that require float input
BenchmarkResult benchmarkCompressorFloat(hshm::Compressor* compressor,
                                         const char* lib_name,
                                         const std::string& distribution,
                                         size_t num_floats) {
    BenchmarkResult result;
    result.library = lib_name;
    result.distribution = distribution;
    result.chunk_size = num_floats * sizeof(float);
    result.success = false;

    // Generate input data (float data type)
    std::vector<float> input_data(num_floats);

    if (distribution == "uniform_float") {
        DataGenerator::generateUniformRandomFloat(input_data.data(), num_floats, distribution);
    } else if (distribution == "normal_float") {
        DataGenerator::generateNormalFloat(input_data.data(), num_floats, distribution);
    } else if (distribution == "repeating_float") {
        DataGenerator::generateRepeatingFloat(input_data.data(), num_floats, distribution);
    } else if (distribution == "structured_float") {
        DataGenerator::generateStructuredFloat(input_data.data(), num_floats, distribution);
    } else if (distribution == "noisy_float") {
        DataGenerator::generateNoisyFloat(input_data.data(), num_floats, distribution);
    } else if (distribution == "random_float") {
        DataGenerator::generateRandomFloat(input_data.data(), num_floats, distribution);
    } else {
        // Unsupported distribution for float data
        return result;
    }

    size_t input_size = num_floats * sizeof(float);

    // Allocate output buffers
    std::vector<uint8_t> compressed_data(input_size * 2);  // Oversized
    std::vector<float> decompressed_data(num_floats);

    // Measure compression
    size_t cmpr_size = compressed_data.size();
    CPUUsage cpu_before = CPUUsage::getCurrent();
    auto start = std::chrono::high_resolution_clock::now();

    bool comp_ok = compressor->Compress(compressed_data.data(), cmpr_size,
                                        input_data.data(), input_size);

    auto end = std::chrono::high_resolution_clock::now();
    CPUUsage cpu_after = CPUUsage::getCurrent();

    if (!comp_ok || cmpr_size == 0) {
        return result;
    }

    auto compress_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    result.compress_time_ms = static_cast<double>(compress_duration.count()) / 1000000.0;

    // Calculate CPU utilization percentage
    if (result.compress_time_ms > 0.0) {
        double cpu_time_ms = cpu_after.cpu_time_ms - cpu_before.cpu_time_ms;
        result.compress_cpu_percent = (cpu_time_ms / result.compress_time_ms) * 100.0;
    } else {
        result.compress_cpu_percent = 0.0;
    }

    // Compression ratio
    if (cmpr_size > 0) {
        result.compression_ratio = static_cast<double>(input_size) / static_cast<double>(cmpr_size);
    } else {
        result.compression_ratio = 0.0;
    }

    // Measure decompression
    size_t decmpr_size = input_size;
    cpu_before = CPUUsage::getCurrent();
    start = std::chrono::high_resolution_clock::now();

    bool decomp_ok = compressor->Decompress(decompressed_data.data(), decmpr_size,
                                            compressed_data.data(), cmpr_size);

    end = std::chrono::high_resolution_clock::now();
    cpu_after = CPUUsage::getCurrent();

    if (!decomp_ok || decmpr_size != input_size) {
        return result;
    }

    auto decompress_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    result.decompress_time_ms = static_cast<double>(decompress_duration.count()) / 1000000.0;

    // Calculate CPU utilization percentage
    if (result.decompress_time_ms > 0.0) {
        double cpu_time_ms = cpu_after.cpu_time_ms - cpu_before.cpu_time_ms;
        result.decompress_cpu_percent = (cpu_time_ms / result.decompress_time_ms) * 100.0;
    } else {
        result.decompress_cpu_percent = 0.0;
    }

    // For lossy compressors, we can't verify exact match
    // Just check that decompression succeeded and produced valid floats
    bool valid = true;
    for (size_t i = 0; i < num_floats; i++) {
        if (!std::isfinite(decompressed_data[i])) {
            valid = false;
            break;
        }
    }

    if (!valid) {
        return result;
    }

    // Calculate quality metrics for lossy compression
    result.mse = QualityMetrics::calculateMSE(input_data.data(), decompressed_data.data(), num_floats);
    result.snr_db = QualityMetrics::calculateSNR(input_data.data(), decompressed_data.data(), num_floats);
    result.psnr_db = QualityMetrics::calculatePSNR(input_data.data(), decompressed_data.data(), num_floats);
    result.max_error = QualityMetrics::calculateMaxError(input_data.data(), decompressed_data.data(), num_floats);

    result.success = true;
    return result;
}

// Print CSV header
void printCSVHeader(std::ostream& os) {
    os << "Library,Distribution,Chunk Size (bytes),"
              << "Compress Time (ms),Decompress Time (ms),"
              << "Compression Ratio,Compress CPU %,Decompress CPU %,"
              << "SNR (dB),PSNR (dB),Max Error,MSE,Success\n";
}

// Print result as CSV
void printResultCSV(const BenchmarkResult& result, std::ostream& os) {
    os << result.library << ","
              << result.distribution << ","
              << result.chunk_size << ","
              << std::fixed << std::setprecision(3) << result.compress_time_ms << ","
              << result.decompress_time_ms << ","
              << std::setprecision(4) << result.compression_ratio << ","
              << std::setprecision(2) << result.compress_cpu_percent << ","
              << result.decompress_cpu_percent << ","
              << std::setprecision(2) << result.snr_db << ","
              << std::setprecision(2) << result.psnr_db << ","
              << std::scientific << std::setprecision(3) << result.max_error << ","
              << result.mse << ","
              << (result.success ? "YES" : "NO") << "\n";
}

TEST_CASE("Lossy Compression Parameter Study") {
    // Fixed chunk size: 64KB for faster testing and focused parameter analysis
    const std::vector<size_t> chunk_sizes = {
        64 * 1024      // 64KB only
    };

    // Floating-point distributions for lossy compressors:
    //
    // FLOAT DISTRIBUTIONS (for lossy compression testing)
    // Organized by compressibility level:
    //
    // HIGHLY COMPRESSIBLE:
    //   repeating_float = smooth repeating sine wave pattern (~50-200x compression)
    //
    // MEDIUM COMPRESSIBLE:
    //   structured_float = smooth signal with small noise (~5-20x compression)
    //   uniform_float = uniform random in [0, 1000] range (~2-10x compression)
    //
    // LIGHTLY COMPRESSIBLE:
    //   noisy_float = weak signal + strong noise (~1.5-5x compression)
    //   normal_float = normal distribution mean=500, stddev=200 (~1.5-5x compression)
    //
    // INCOMPRESSIBLE:
    //   random_float = pure white noise (~1.0-1.2x compression)
    const std::vector<std::string> float_distributions = {
        "repeating_float",    // Highly compressible
        "structured_float",   // Medium compressible
        "uniform_float",      // Medium compressible
        "noisy_float",        // Lightly compressible
        "normal_float",       // Lightly compressible
        "random_float"        // Incompressible
    };

    // Open output file for lossy compression results
    std::ofstream outfile("compression_lossy_parameter_study_results.csv");
    if (!outfile.is_open()) {
        std::cerr << "Warning: Could not open output file. Results will only be printed to console.\n";
    }

    // Print headers to both console and file
    printCSVHeader(std::cout);
    if (outfile.is_open()) {
        printCSVHeader(outfile);
    }

    // Test each compression library
    struct CompressorTest {
        std::string name;
        std::unique_ptr<hshm::Compressor> compressor;
    };

    // Lossy compressors for floating-point data - PARAMETER STUDY
    // Test multiple parameter values for each compressor
#if HSHM_ENABLE_COMPRESS
    std::vector<CompressorTest> lossy_compressors;

    // ZFP: Test multiple error tolerances
    std::vector<double> zfp_tolerances = {1e-1, 1e-2, 1e-3, 1e-4, 1e-5};
    for (double tol : zfp_tolerances) {
        std::string name = "ZFP_tol_" + std::to_string(tol);
        lossy_compressors.push_back({name, std::make_unique<hshm::Zfp>(tol)});
    }

    // BitGrooming: Test multiple significant digits
    std::vector<int> bitgrooming_nsds = {1, 2, 3, 4, 5};
    for (int nsd : bitgrooming_nsds) {
        std::string name = "BitGrooming_nsd_" + std::to_string(nsd);
        lossy_compressors.push_back({name, std::make_unique<hshm::BitGrooming>(nsd)});
    }

    // FPZIP: Test multiple precision levels
    std::vector<int> fpzip_precisions = {8, 12, 16, 20, 24, 0};  // 0 = lossless
    for (int prec : fpzip_precisions) {
        std::string name = "FPZIP_prec_" + std::to_string(prec);
        lossy_compressors.push_back({name, std::make_unique<hshm::Fpzip>(prec)});
    }

    // SZ3 and SZ: Commented out due to header/API compatibility issues
    // Can be enabled once proper headers are installed
    // std::vector<double> sz_tolerances = {1e-1, 1e-2, 1e-3, 1e-4, 1e-5};
    // for (double tol : sz_tolerances) {
    //     std::string name = "SZ3_tol_" + std::to_string(tol);
    //     lossy_compressors.push_back({name, std::make_unique<hshm::Sz3>(tol, hshm::Sz3::ErrorMode::ABS)});
    // }
#endif
    // Test lossy compressors with floating-point data
    // These compressors require float input and support lossy compression
    for (const auto& test : lossy_compressors) {
        const std::string& lib_name = test.name;
        hshm::Compressor* compressor = test.compressor.get();

        std::cerr << "Starting benchmark for: " << lib_name << " (float data)" << std::endl;
        std::cout.flush();

        try {
            for (const auto& distribution : float_distributions) {
                for (size_t chunk_size : chunk_sizes) {
                    // Calculate number of floats for the given byte size
                    size_t num_floats = chunk_size / sizeof(float);

                    std::cerr << "  Testing: " << distribution
                              << ", " << (chunk_size/1024) << "KB (" << num_floats << " floats)" << std::endl;

                    auto result = benchmarkCompressorFloat(compressor, lib_name.c_str(),
                                                          distribution, num_floats);

                    // Print to both console and file
                    printResultCSV(result, std::cout);
                    if (outfile.is_open()) {
                        printResultCSV(result, outfile);
                    }
                    std::cout.flush();
                    if (outfile.is_open()) {
                        outfile.flush();
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "ERROR in " << lib_name << ": " << e.what() << std::endl;
        }

        std::cerr << "Completed benchmark for: " << lib_name << std::endl;
    }

    outfile.close();
    std::cout << "\nResults saved to: compression_lossy_parameter_study_results.csv" << std::endl;
}
