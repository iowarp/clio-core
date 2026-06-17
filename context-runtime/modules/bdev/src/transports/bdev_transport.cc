/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

#include <clio_runtime/bdev/transports/bdev_transport.h>
#include <clio_runtime/bdev/transports/fs_bdev_transport.h>
#include <clio_runtime/bdev/transports/mem_bdev_transport.h>
#include <clio_runtime/bdev/transports/s3_bdev_transport.h>

namespace clio::run::bdev {

std::unique_ptr<BdevTransport> BdevTransportFactory::Create(BdevType type) {
  switch (type) {
    case BdevType::kFile:
      return std::make_unique<FsBdevTransport>();
    case BdevType::kRam:
      return std::make_unique<MemBdevTransport>();
    // Assume kPinned or kHbm are currently using RAM path based on original bdev_runtime.cc
    case BdevType::kHbm:
    case BdevType::kPinned:
      return std::make_unique<MemBdevTransport>(); 
    case BdevType::kS3:
      return std::make_unique<S3BdevTransport>();
    default:
      return nullptr;
  }
}

} // namespace clio::run::bdev
