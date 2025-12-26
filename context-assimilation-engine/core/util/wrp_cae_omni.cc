/**
 * wrp_cae_ingest - Ingest OMNI file for CAE processing
 *
 * This utility reads an OMNI YAML file and calls ParseOmni to schedule
 * assimilation tasks. Usage: wrp_cae_ingest <omni_file_path>
 */

#include <hermes_shm/util/config_parse.h>
#include <wrp_cae/core/constants.h>
#include <wrp_cae/core/core_client.h>
#include <wrp_cae/core/factory/assimilation_ctx.h>
#include <yaml-cpp/yaml.h>

#include <iostream>
#include <string>
#include <vector>

/**
 * Load OMNI configuration file and produce vector of AssimilationCtx
 */
std::vector<wrp_cae::core::AssimilationCtx> LoadOmni(
    const std::string& omni_path) {
  std::cout << "Loading OMNI file: " << omni_path << std::endl;

  YAML::Node config;
  try {
    config = YAML::LoadFile(omni_path);
  } catch (const YAML::Exception& e) {
    throw std::runtime_error("Failed to load OMNI file: " +
                             std::string(e.what()));
  }

  // Check for required 'transfers' key
  if (!config["transfers"]) {
    throw std::runtime_error("OMNI file missing required 'transfers' key");
  }

  const YAML::Node& transfers = config["transfers"];
  if (!transfers.IsSequence()) {
    throw std::runtime_error("OMNI 'transfers' must be a sequence/array");
  }

  std::vector<wrp_cae::core::AssimilationCtx> contexts;
  contexts.reserve(transfers.size());

  // Parse each transfer entry
  for (size_t i = 0; i < transfers.size(); ++i) {
    const YAML::Node& transfer = transfers[i];

    // Validate required fields
    if (!transfer["src"]) {
      throw std::runtime_error("Transfer " + std::to_string(i + 1) +
                               " missing required 'src' field");
    }
    if (!transfer["dst"]) {
      throw std::runtime_error("Transfer " + std::to_string(i + 1) +
                               " missing required 'dst' field");
    }
    if (!transfer["format"]) {
      throw std::runtime_error("Transfer " + std::to_string(i + 1) +
                               " missing required 'format' field");
    }

    wrp_cae::core::AssimilationCtx ctx;
    ctx.src = transfer["src"].as<std::string>();
    ctx.dst = transfer["dst"].as<std::string>();
    ctx.format = transfer["format"].as<std::string>();
    ctx.depends_on =
        transfer["depends_on"] ? transfer["depends_on"].as<std::string>() : "";
    ctx.range_off =
        transfer["range_off"] ? transfer["range_off"].as<size_t>() : 0;
    ctx.range_size =
        transfer["range_size"] ? transfer["range_size"].as<size_t>() : 0;

    // Parse tokens and expand environment variables
    if (transfer["src_token"]) {
      std::string raw_token = transfer["src_token"].as<std::string>();
      ctx.src_token = hshm::ConfigParse::ExpandPath(raw_token);
    }
    if (transfer["dst_token"]) {
      std::string raw_token = transfer["dst_token"].as<std::string>();
      ctx.dst_token = hshm::ConfigParse::ExpandPath(raw_token);
    }

    // Parse dataset_filter for HDF5 and other hierarchical formats
    if (transfer["dataset_filter"]) {
      const YAML::Node& filter = transfer["dataset_filter"];

      // Parse include_patterns
      if (filter["include_patterns"]) {
        const YAML::Node& include_node = filter["include_patterns"];
        if (include_node.IsSequence()) {
          for (size_t j = 0; j < include_node.size(); ++j) {
            ctx.include_patterns.push_back(include_node[j].as<std::string>());
          }
        }
      }

      // Parse exclude_patterns
      if (filter["exclude_patterns"]) {
        const YAML::Node& exclude_node = filter["exclude_patterns"];
        if (exclude_node.IsSequence()) {
          for (size_t j = 0; j < exclude_node.size(); ++j) {
            ctx.exclude_patterns.push_back(exclude_node[j].as<std::string>());
          }
        }
      }
    }

    contexts.push_back(ctx);

    std::cout << "  Loaded transfer " << (i + 1) << "/" << transfers.size()
              << ":" << std::endl;
    std::cout << "    src: " << ctx.src << std::endl;
    std::cout << "    dst: " << ctx.dst << std::endl;
    std::cout << "    format: " << ctx.format << std::endl;
    if (!ctx.src_token.empty()) {
      std::cout << "    src_token: <set>" << std::endl;
    }
    if (!ctx.dst_token.empty()) {
      std::cout << "    dst_token: <set>" << std::endl;
    }
    if (!ctx.include_patterns.empty()) {
      std::cout << "    dataset_filter.include_patterns: [";
      for (size_t j = 0; j < ctx.include_patterns.size(); ++j) {
        if (j > 0) std::cout << ", ";
        std::cout << "\"" << ctx.include_patterns[j] << "\"";
      }
      std::cout << "]" << std::endl;
    }
    if (!ctx.exclude_patterns.empty()) {
      std::cout << "    dataset_filter.exclude_patterns: [";
      for (size_t j = 0; j < ctx.exclude_patterns.size(); ++j) {
        if (j > 0) std::cout << ", ";
        std::cout << "\"" << ctx.exclude_patterns[j] << "\"";
      }
      std::cout << "]" << std::endl;
    }
  }

  std::cout << "Successfully loaded " << contexts.size()
            << " transfer(s) from OMNI file" << std::endl;
  return contexts;
}

void PrintUsage(const char* program_name) {
  std::cerr << "Usage: " << program_name << " <omni_file_path>" << std::endl;
  std::cerr << "  omni_file_path - Path to the OMNI YAML file to ingest"
            << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    PrintUsage(argv[0]);
    return 1;
  }

  std::string omni_file_path(argv[1]);

  try {
    // Initialize Chimaera client
    if (!chi::CHIMAERA_INIT(chi::ChimaeraMode::kClient, false)) {
      std::cerr << "Error: Failed to initialize Chimaera client" << std::endl;
      return 1;
    }

    // Verify Chimaera IPC is available
    auto* ipc_manager = CHI_IPC;
    if (!ipc_manager) {
      std::cerr
          << "Error: Chimaera IPC not initialized. Is the runtime running?"
          << std::endl;
      return 1;
    }

    // Load OMNI file and parse transfers
    std::vector<wrp_cae::core::AssimilationCtx> contexts =
        LoadOmni(omni_file_path);

    // Connect to CAE core container using the standard pool ID
    wrp_cae::core::Client client(wrp_cae::core::kCaePoolId);

    std::cout << "Calling ParseOmni..." << std::endl;
    std::cout.flush();

    // Call ParseOmni with vector of contexts
    chi::u32 num_tasks_scheduled = 0;
    chi::u32 result = client.ParseOmni(contexts, num_tasks_scheduled);

    if (result != 0) {
      std::cerr << "Error: ParseOmni failed with result code " << result
                << std::endl;
      return 1;
    }

    std::cout << "ParseOmni completed successfully!" << std::endl;
    std::cout << "  Tasks scheduled: " << num_tasks_scheduled << std::endl;

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
