/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 */

#include <clio_runtime/bdev/transports/bdev_transport.h>
#include <clio_runtime/bdev/transports/fs_bdev_transport.h>
#include <clio_runtime/bdev/transports/mem_bdev_transport.h>

#ifdef CLIO_ENABLE_AMAZON_DRIVE
#include <clio_runtime/bdev/transports/s3_bdev_transport.h>
#endif

#ifdef CLIO_ENABLE_GOOGLE_CLOUD
#include <clio_runtime/bdev/transports/gcs_bdev_transport.h>
#endif

namespace clio::run::bdev {

std::unique_ptr<BdevTransport> BdevTransportFactory::Create(BdevType type) {
  switch (type) {
    case BdevType::kFile:
      return std::make_unique<FsBdevTransport>();
    case BdevType::kRam:
    case BdevType::kHbm:
    case BdevType::kPinned:
      return std::make_unique<MemBdevTransport>(); 
#ifdef CLIO_ENABLE_AMAZON_DRIVE
    case BdevType::kS3:
      return std::make_unique<S3BdevTransport>();
#endif
#ifdef CLIO_ENABLE_GOOGLE_CLOUD
    case BdevType::kGcs:
      return std::make_unique<GcsBdevTransport>();
#endif
    default:
      return nullptr;
  }
}

} // namespace clio::run::bdev
