#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "chimaera/chimaera.h"
#include "chimaera/admin/admin_client.h"
#include "chimaera/types.h"
#include "chimaera_commands.h"

namespace {
void PrintMigrateUsage() {
  HIPRINT("Usage: chimaera migrate --pool-id <major.minor> --container-id <CID> --node-id <NID>");
  HIPRINT("  Migrate a container to a different node");
  HIPRINT("");
  HIPRINT("Options:");
  HIPRINT("  --pool-id <major.minor>  Pool ID (e.g., 200.0)");
  HIPRINT("  --container-id <CID>     Container ID to migrate");
  HIPRINT("  --node-id <NID>          Destination node ID");
}
}  // namespace

int Migrate(int argc, char** argv) {
  std::string pool_id_str;
  chi::u32 container_id = 0;
  chi::u32 node_id = 0;
  bool has_pool = false, has_container = false, has_node = false;

  for (int i = 0; i < argc; ++i) {
    if (std::strcmp(argv[i], "--pool-id") == 0 && i + 1 < argc) {
      pool_id_str = argv[++i];
      has_pool = true;
    } else if (std::strcmp(argv[i], "--container-id") == 0 && i + 1 < argc) {
      container_id = static_cast<chi::u32>(std::atoi(argv[++i]));
      has_container = true;
    } else if (std::strcmp(argv[i], "--node-id") == 0 && i + 1 < argc) {
      node_id = static_cast<chi::u32>(std::atoi(argv[++i]));
      has_node = true;
    } else if (std::strcmp(argv[i], "--help") == 0 ||
               std::strcmp(argv[i], "-h") == 0) {
      PrintMigrateUsage();
      return 0;
    }
  }

  if (!has_pool || !has_container || !has_node) {
    HLOG(kError, "Missing required arguments");
    PrintMigrateUsage();
    return 1;
  }

  // Parse pool ID from "major.minor" format
  chi::PoolId pool_id = chi::UniqueId::FromString(pool_id_str);

  if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, false)) {
    HLOG(kError, "Failed to initialize Chimaera client");
    return 1;
  }

  auto* admin_client = CHI_ADMIN;
  if (!admin_client) {
    HLOG(kError, "Failed to get admin client");
    return 1;
  }

  // Build migration request
  std::vector<chi::MigrateInfo> migrations;
  migrations.emplace_back(pool_id, container_id, node_id);

  HLOG(kInfo, "Migrating pool {} container {} to node {}",
       pool_id, container_id, node_id);

  auto task = admin_client->AsyncMigrateContainers(
      chi::PoolQuery::Local(), migrations);
  task.Wait();

  if (task->GetReturnCode() != 0) {
    HLOG(kError, "Migration failed: {}", task->error_message_.str());
    return 1;
  }

  HLOG(kSuccess, "Successfully migrated {} container(s)", task->num_migrated_);
  return 0;
}
