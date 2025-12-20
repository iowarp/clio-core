/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
* Distributed under BSD 3-Clause license.                                   *
* Copyright by The HDF Group.                                               *
* Copyright by the Illinois Institute of Technology.                        *
* All rights reserved.                                                      *
*                                                                           *
* This file is part of Hermes. The full Hermes copyright notice, including  *
* terms governing use, modification, and redistribution, is contained in    *
* the COPYING file, which can be found at the top directory. If you do not  *
* have access to the file, you may request a copy from help@hdfgroup.org.   *
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "basic_test.h"
#include "hermes_shm/util/compress/compress_factory.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>
#include <algorithm>

#ifdef __linux__
#include <sys/resource.h>
#include <unistd.h>
#endif

#if HSHM_ENABLE_COMPRESS
#include <lzo/lzo1x.h>
#endif

struct BenchmarkResult {
    std::string library;
    std::string data_type;
    std::string distribution;
    size_t chunk_size;
    double compress_time_ms;
    double decompress_time_ms;
    double compression_ratio;
    double compress_cpu_percent;
    double decompress_cpu_percent;
    bool success;
};

// CPU usage tracking
struct CPUUsage {
    double user_time;
    double system_time;
    
#ifdef __linux__
    static CPUUsage getCurrent() {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        return {
            usage.ru_utime.tv_sec * 1000.0 + usage.ru_utime.tv_usec / 1000.0,
            usage.ru_stime.tv_sec * 1000.0 + usage.ru_stime.tv_usec / 1000.0
        };
    }
#else
    static CPUUsage getCurrent() {
        return {0.0, 0.0};
    }
#endif
};

// Data distribution generators
class DataGenerator {
public:
    static void generateUniformRandom(void* data, size_t size, size_t type_size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        uint8_t* bytes = static_cast<uint8_t*>(data);
        for (size_t i = 0; i < size * type_size; i++) {
            bytes[i] = gen() & 0xFF;
        }
    }
    
    static void generateNormal(void* data, size_t size, size_t type_size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(128.0, 30.0); // Mean=128, StdDev=30
        
        if (type_size == 1) {
            uint8_t* bytes = static_cast<uint8_t*>(data);
            for (size_t i = 0; i < size; i++) {
                bytes[i] = static_cast<uint8_t>(std::clamp(dist(gen), 0.0, 255.0));
            }
        } else if (type_size == 4) {
            float* floats = static_cast<float*>(data);
            std::normal_distribution<float> fdist(0.0f, 1.0f);
            for (size_t i = 0; i < size; i++) {
                floats[i] = fdist(gen);
            }
        } else if (type_size == 8) {
            double* doubles = static_cast<double*>(data);
            std::normal_distribution<double> ddist(0.0, 1.0);
            for (size_t i = 0; i < size; i++) {
                doubles[i] = ddist(gen);
            }
        }
    }
    
    static void generateGamma(void* data, size_t size, size_t type_size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::gamma_distribution<> dist(2.0, 2.0); // Shape=2, Scale=2
        
        if (type_size == 1) {
            uint8_t* bytes = static_cast<uint8_t*>(data);
            for (size_t i = 0; i < size; i++) {
                bytes[i] = static_cast<uint8_t>(std::clamp(dist(gen) * 10, 0.0, 255.0));
            }
        } else if (type_size == 4) {
            float* floats = static_cast<float*>(data);
            std::gamma_distribution<float> fdist(2.0f, 2.0f);
            for (size_t i = 0; i < size; i++) {
                floats[i] = fdist(gen);
            }
        } else if (type_size == 8) {
            double* doubles = static_cast<double*>(data);
            std::gamma_distribution<double> ddist(2.0, 2.0);
            for (size_t i = 0; i < size; i++) {
                doubles[i] = ddist(gen);
            }
        }
    }
    
    static void generateExponential(void* data, size_t size, size_t type_size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::exponential_distribution<> dist(1.0); // Lambda=1.0
        
        if (type_size == 1) {
            uint8_t* bytes = static_cast<uint8_t*>(data);
            for (size_t i = 0; i < size; i++) {
                bytes[i] = static_cast<uint8_t>(std::clamp(dist(gen) * 50, 0.0, 255.0));
            }
        } else if (type_size == 4) {
            float* floats = static_cast<float*>(data);
            std::exponential_distribution<float> fdist(1.0f);
            for (size_t i = 0; i < size; i++) {
                floats[i] = fdist(gen);
            }
        } else if (type_size == 8) {
            double* doubles = static_cast<double*>(data);
            std::exponential_distribution<double> ddist(1.0);
            for (size_t i = 0; i < size; i++) {
                doubles[i] = ddist(gen);
            }
        }
    }
    
    static void generateRepeating(void* data, size_t size, size_t type_size) {
        // Create highly compressible data
        uint8_t pattern[] = {0xAA, 0x55, 0xAA, 0x55};
        uint8_t* bytes = static_cast<uint8_t*>(data);
        for (size_t i = 0; i < size * type_size; i++) {
            bytes[i] = pattern[i % sizeof(pattern)];
        }
    }
};

// Benchmark a single compression library
BenchmarkResult benchmarkCompressor(hshm::Compressor* compressor, 
                                   const std::string& lib_name,
                                   const std::string& data_type,
                                   const std::string& distribution,
                                   size_t chunk_size,
                                   size_t type_size) {
    BenchmarkResult result;
    result.library = lib_name;
    result.data_type = data_type;
    result.distribution = distribution;
    result.chunk_size = chunk_size;
    result.success = false;
    
    // Allocate buffers
    std::vector<uint8_t> input_data(chunk_size);
    std::vector<uint8_t> compressed_data(chunk_size * 2); // Safety margin
    std::vector<uint8_t> decompressed_data(chunk_size);
    
    // Generate test data
    if (distribution == "uniform") {
        DataGenerator::generateUniformRandom(input_data.data(), chunk_size / type_size, type_size);
    } else if (distribution == "normal") {
        DataGenerator::generateNormal(input_data.data(), chunk_size / type_size, type_size);
    } else if (distribution == "gamma") {
        DataGenerator::generateGamma(input_data.data(), chunk_size / type_size, type_size);
    } else if (distribution == "exponential") {
        DataGenerator::generateExponential(input_data.data(), chunk_size / type_size, type_size);
    } else if (distribution == "repeating") {
        DataGenerator::generateRepeating(input_data.data(), chunk_size / type_size, type_size);
    }
    
    // Warmup - reset compressed buffer size before warmup
    size_t warmup_cmpr_size = compressed_data.size();
    size_t warmup_decmpr_size = decompressed_data.size();
    bool warmup_comp_ok = compressor->Compress(compressed_data.data(), warmup_cmpr_size, 
                        input_data.data(), chunk_size);
    if (!warmup_comp_ok) {
        // Warmup failed, skip this test
        return result;
    }
    bool warmup_decomp_ok = compressor->Decompress(decompressed_data.data(), warmup_decmpr_size,
                          compressed_data.data(), warmup_cmpr_size);
    if (!warmup_decomp_ok) {
        // Warmup decompression failed, skip this test
        return result;
    }
    
    // Reset buffer sizes for actual measurement
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
    
    auto compress_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    result.compress_time_ms = compress_duration.count() / 1000.0;
    
    // Safety check for division by zero
    if (result.compress_time_ms > 0.0) {
    double cpu_time_ms = (cpu_after.user_time + cpu_after.system_time) - 
                        (cpu_before.user_time + cpu_before.system_time);
    result.compress_cpu_percent = (cpu_time_ms / result.compress_time_ms) * 100.0;
    } else {
        result.compress_cpu_percent = 0.0;
    }
    
    // Compression ratio: original_size / compressed_size (higher is better)
    // Safety check for division by zero
    if (cmpr_size > 0) {
        result.compression_ratio = static_cast<double>(chunk_size) / cmpr_size;
    } else {
        result.compression_ratio = 0.0;
    }
    
    // Measure decompression
    size_t decmpr_size = decompressed_data.size();
    cpu_before = CPUUsage::getCurrent();
    start = std::chrono::high_resolution_clock::now();
    
    bool decomp_ok = compressor->Decompress(decompressed_data.data(), decmpr_size,
                                           compressed_data.data(), cmpr_size);
    
    end = std::chrono::high_resolution_clock::now();
    cpu_after = CPUUsage::getCurrent();
    
    if (!decomp_ok || decmpr_size != chunk_size) {
        return result;
    }
    
    auto decompress_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    result.decompress_time_ms = decompress_duration.count() / 1000.0;
    
    // Safety check for division by zero
    if (result.decompress_time_ms > 0.0) {
        double cpu_time_ms = (cpu_after.user_time + cpu_after.system_time) - 
                  (cpu_before.user_time + cpu_before.system_time);
    result.decompress_cpu_percent = (cpu_time_ms / result.decompress_time_ms) * 100.0;
    } else {
        result.decompress_cpu_percent = 0.0;
    }
    
    // Verify correctness
    result.success = (memcmp(input_data.data(), decompressed_data.data(), chunk_size) == 0);
    
    return result;
}

// Print CSV table header
void printCSVHeader(std::ostream& os) {
    os << "Library,Data Type,Distribution,Chunk Size (bytes),"
              << "Compress Time (ms),Decompress Time (ms),"
              << "Compression Ratio,Compress CPU %,Decompress CPU %,Success\n";
}

// Print result as CSV
void printResultCSV(const BenchmarkResult& result, std::ostream& os) {
    os << result.library << ","
              << result.data_type << ","
              << result.distribution << ","
              << result.chunk_size << ","
              << std::fixed << std::setprecision(3) << result.compress_time_ms << ","
              << result.decompress_time_ms << ","
              << std::setprecision(4) << result.compression_ratio << ","
              << std::setprecision(2) << result.compress_cpu_percent << ","
              << result.decompress_cpu_percent << ","
              << (result.success ? "YES" : "NO") << "\n";
}

TEST_CASE("Compression Benchmark") {
    const std::vector<size_t> chunk_sizes = {
        64 * 1024,      // 64KB
        128 * 1024,     // 128KB
        1024 * 1024,    // 1MB
        4 * 1024 * 1024,  // 4MB
        16 * 1024 * 1024  // 16MB
        // 32MB removed to reduce benchmark time
    };
    
    const std::vector<std::string> distributions = {
        "uniform", "normal", "gamma", "exponential", "repeating"
    };
    
    const std::vector<std::pair<std::string, size_t>> data_types = {
        {"char", 1},
        {"int", 4},
        {"int64", 8},
        {"float", 4},
        {"double", 8}
    };
    
    // Open output file
    std::ofstream outfile("compression_benchmark_results.csv");
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
    
    // Initialize LZO library (required before use)
#if HSHM_ENABLE_COMPRESS
    lzo_init();
    // Note: LZO is currently skipped due to crashes with large buffers
    // The LZO wrapper may need work memory allocation (see lzo.h)
#endif
    
    std::vector<CompressorTest> compressors;
    compressors.push_back({"BZIP2", std::make_unique<hshm::Bzip2>()});
    compressors.push_back({"LZO", std::make_unique<hshm::Lzo>()});
    compressors.push_back({"Zstd", std::make_unique<hshm::Zstd>()});
    compressors.push_back({"LZ4", std::make_unique<hshm::Lz4>()});
    compressors.push_back({"Zlib", std::make_unique<hshm::Zlib>()});
    compressors.push_back({"Lzma", std::make_unique<hshm::Lzma>()});
    compressors.push_back({"Brotli", std::make_unique<hshm::Brotli>()});
    compressors.push_back({"Snappy", std::make_unique<hshm::Snappy>()});
    compressors.push_back({"Blosc2", std::make_unique<hshm::Blosc>()});
    
    for (const auto& test : compressors) {
        const std::string& lib_name = test.name;
        hshm::Compressor* compressor = test.compressor.get();
        
        std::cerr << "Starting benchmark for: " << lib_name << std::endl;
        std::cout.flush();
        
        try {
        for (const auto& [data_type, type_size] : data_types) {
            for (const auto& distribution : distributions) {
                for (size_t chunk_size : chunk_sizes) {
                    // Skip if chunk size is not aligned with data type
                    if (chunk_size % type_size != 0) {
                        continue;
                    }
                    
                        std::cerr << "  Testing: " << data_type << ", " << distribution 
                                  << ", " << (chunk_size/1024) << "KB" << std::endl;
                        
                    auto result = benchmarkCompressor(compressor, lib_name, 
                                                     data_type, distribution,
                                                     chunk_size, type_size);
                    
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
            }
        } catch (const std::exception& e) {
            std::cerr << "Error benchmarking " << lib_name << ": " << e.what() << std::endl;
            continue;
        } catch (...) {
            std::cerr << "Unknown error benchmarking " << lib_name << std::endl;
            continue;
        }
        
        std::cerr << "Completed benchmark for: " << lib_name << std::endl;
        std::cout.flush();
    }
    
    if (outfile.is_open()) {
        outfile.close();
        std::cout << "\nResults saved to: compression_benchmark_results.csv\n";
    }
}

