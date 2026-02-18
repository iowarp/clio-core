from jarvis_cd.core.pkg import Service
from jarvis_cd.util import SizeType
from jarvis_cd.shell.process import Rm, Mkdir
from jarvis_cd.shell import PsshExecInfo, Exec, LocalExecInfo
import yaml
import os

class WrpCte(Service):
    """
    Content Transfer Engine (CTE) configuration service for IoWarp.

    This service configures the CTE core module by generating a chimaera compose
    compatible YAML configuration file. The start() method uses chimaera compose
    with PsshExecInfo to deploy the CTE across all nodes.
    """

    def _init(self):
        """
        Initialize the WrpCte service.

        This method is called during service initialization.
        """
        self.compose_config_path = os.path.join(self.shared_dir, 'cte_compose.yaml')
        self.devices_from_resource_graph = []

    def _configure_menu(self):
        """
        Configure the service menu.

        Returns:
            List[Dict]: Configuration menu options for CTE.
        """
        return [
            {
                'name': 'pool_name',
                'msg': 'Name of the CTE pool',
                'type': str,
                'default': 'wrp_cte'
            },
            {
                'name': 'pool_id',
                'msg': 'Pool ID for the CTE pool',
                'type': float,
                'default': 512.0
            },
            {
                'name': 'pool_query',
                'msg': 'Pool query type (local or dynamic)',
                'type': str,
                'choices': ['local', 'dynamic'],
                'default': 'local'
            },
            {
                'name': 'devices',
                'msg': 'List of storage devices as tuples (path, capacity, score)',
                'type': list,
                'default': [],
                'help': 'Example: [("/mnt/nvme", "1TB", 0.9), ("ram::cache", "8GB", 1.0)]. Use ram:: prefix for RAM storage. Supports SizeType format: k/K, m/M, g/G, t/T'
            },
            {
                'name': 'dpe_type',
                'msg': 'Data Placement Engine type',
                'type': str,
                'choices': ['random', 'round_robin', 'max_bw'],
                'default': 'max_bw'
            },
            {
                'name': 'neighborhood',
                'msg': 'Number of targets (nodes CTE can buffer to)',
                'type': int,
                'default': 4
            },
            {
                'name': 'default_target_timeout_ms',
                'msg': 'Default target timeout in milliseconds',
                'type': int,
                'default': 30000
            },
            {
                'name': 'poll_period_ms',
                'msg': 'Period at which targets should be rescanned for statistics (capacity, bandwidth, etc.) in milliseconds',
                'type': int,
                'default': 5000
            },
            {
                'name': 'monitor_interval_ms',
                'msg': 'Compression monitor interval for collecting target capacities and stats (milliseconds)',
                'type': int,
                'default': 5
            },
            {
                'name': 'dnn_model_weights_path',
                'msg': 'Path to DNN model weights JSON file for compression prediction (empty = disabled)',
                'type': str,
                'default': ''
            },
            {
                'name': 'dnn_samples_before_reinforce',
                'msg': 'Number of samples to collect before reinforcing DNN model',
                'type': int,
                'default': 1000
            },
            {
                'name': 'trace_folder_path',
                'msg': 'Path to folder for CTE trace logs (empty = disabled)',
                'type': str,
                'default': ''
            },
            {
                'name': 'iowarp_compress',
                'msg': 'Compression mode for IOWarp Engine',
                'type': str,
                'default': 'none',
                'help': 'Environment variable IOWARP_COMPRESS: none/off (no compression), dynamic/auto (adaptive), or library name (zstd, lz4, brotli, bzip2, blosc2, fpzip, lzma, snappy, sz3, zfp, zlib)'
            },
            {
                'name': 'iowarp_compress_trace',
                'msg': 'Enable compression tracing',
                'type': str,
                'choices': ['on', 'off'],
                'default': 'off',
                'help': 'Environment variable IOWARP_COMPRESS_TRACE: on/1/true (enable tracing), off (disable)'
            }
        ]

    def _configure(self, **kwargs):
        """
        Configure the CTE service with provided keyword arguments.

        This method generates a chimaera compose compatible YAML configuration
        file that will be used by the start() method to deploy CTE.

        Args:
            **kwargs: Configuration arguments from _configure_menu.
        """
        self.log("Configuring Content Transfer Engine (CTE)...")

        # Handle devices configuration
        devices = self.config.get('devices', [])

        if not devices:
            self.log("No devices specified, attempting to use resource graph...")
            devices = self._get_devices_from_resource_graph()
            if not devices:
                self.log("Warning: No devices available from resource graph, using defaults")
                devices = self._get_default_devices()
        else:
            # Validate and convert device tuples
            devices = self._validate_and_convert_devices(devices)

        # Build chimaera compose compatible configuration
        compose_config = self._build_compose_config(devices)

        try:
            with open(self.compose_config_path, 'w') as f:
                f.write('# Content Transfer Engine (CTE) Compose Configuration\n')
                f.write('# Generated by Jarvis WrpCte Package\n\n')
                yaml.dump(compose_config, f, default_flow_style=False, indent=2)
            self.log(f"CTE compose configuration written to: {self.compose_config_path}")
        except Exception as e:
            self.log(f"Error writing CTE compose configuration: {e}")
            raise

        # Create parent directories for each device (skip RAM devices)
        self.log("Creating parent directories for storage devices...")
        for path, _, _ in devices:
            # Skip RAM devices (they don't need directories)
            if path.startswith('ram::'):
                continue
            parent_dir = os.path.dirname(path)
            if parent_dir:
                self.log(f"Creating directory: {parent_dir}")
                try:
                    Mkdir(parent_dir, PsshExecInfo(hostfile=self.hostfile)).run()
                    self.log(f"Created directory: {parent_dir}")
                except Exception as e:
                    self.log(f"Error creating directory {parent_dir}: {e}")

        self.log("CTE configuration completed successfully")

    def _get_devices_from_resource_graph(self):
        """
        Extract device information from the resource graph.

        Returns:
            List[Tuple[str, str, float]]: List of device tuples (path, capacity, score).
        """
        try:
            from jarvis_cd.core.resource_graph import ResourceGraphManager

            # Initialize ResourceGraphManager (gets Jarvis singleton internally)
            rg_manager = ResourceGraphManager()

            if not rg_manager.resource_graph.get_all_nodes():
                self.log("Warning: Resource graph is empty. Run 'jarvis rg build' first.")
                return []

            # Get common storage - returns dict mapping mount points to device lists
            common_storage = rg_manager.resource_graph.get_common_storage()
            if not common_storage:
                self.log("Warning: No common storage found in resource graph")
                return []

            devices = []
            # Iterate over common storage mount points
            for _, device_list in common_storage.items():
                # Process each device at this mount point
                for device in device_list:
                    # Use mount point as base path
                    base_path = device.get('mount', '/tmp/cte_storage')
                    # Append hermes_data.bin for bdev file creation
                    path = os.path.join(base_path, 'hermes_data.bin')

                    # Get available space and multiply by 0.5 for safety margin
                    available_space = device.get('avail', '100GB')
                    capacity = self._adjust_capacity(available_space, 0.5)

                    # Calculate score based on device type
                    device_type = device.get('dev_type', 'unknown').lower()
                    score = self._calculate_device_score(device_type, device)

                    devices.append((path, capacity, score))
                    self.log(f"Found storage device: {path} (available: {available_space}, using: {capacity}, score: {score}, type: {device_type})")

            return devices

        except ImportError as e:
            self.log(f"Warning: ResourceGraphManager not available: {e}")
            return []
        except Exception as e:
            self.log(f"Warning: Error accessing resource graph: {e}")
            return []

    def _adjust_capacity(self, capacity_str, factor):
        """
        Adjust capacity by a factor (e.g., multiply by 0.5 for safety margin).

        Args:
            capacity_str (str): Capacity string (e.g., "100GB", "1TB")
            factor (float): Multiplication factor

        Returns:
            str: Adjusted capacity string in same format
        """
        import re

        # Parse capacity string
        match = re.match(r'([\d.]+)\s*([KMGT]?B?)', capacity_str.upper().strip())
        if not match:
            # If parsing fails, return original
            return capacity_str

        value = float(match.group(1))
        suffix = match.group(2)

        # Apply factor
        adjusted_value = value * factor

        # Format back to string with 2 decimal places if needed
        if adjusted_value == int(adjusted_value):
            return f"{int(adjusted_value)}{suffix}"
        else:
            return f"{adjusted_value:.2f}{suffix}"

    def _calculate_device_score(self, storage_type, storage_info):
        """
        Calculate a performance score for a storage device.

        Args:
            storage_type (str): Type of storage (nvme, ssd, hdd, ram, etc.)
            storage_info (dict): Additional storage information from resource graph.

        Returns:
            float: Score between 0.0 and 1.0.
        """
        # Base scores by storage type
        type_scores = {
            'ram': 1.0,
            'ramdisk': 1.0,
            'tmpfs': 1.0,
            'nvme': 0.9,
            'ssd': 0.7,
            'hdd': 0.4,
            'network': 0.3,
            'unknown': 0.5
        }

        base_score = type_scores.get(storage_type, 0.5)

        # Adjust based on performance benchmarks from resource graph
        # Resource graph provides '4k_randwrite_bw' and '1m_seqwrite_bw' if benchmarked
        seq_bw = storage_info.get('1m_seqwrite_bw', 'unknown')
        rand_bw = storage_info.get('4k_randwrite_bw', 'unknown')

        # Boost score for high sequential bandwidth
        if seq_bw != 'unknown' and isinstance(seq_bw, str):
            try:
                # Parse bandwidth strings like "500MB/s" or "1.5GB/s"
                if 'GB/s' in seq_bw:
                    bw_value = float(seq_bw.replace('GB/s', '').strip())
                    base_score = min(1.0, base_score + 0.2)  # High performance boost
                elif 'MB/s' in seq_bw:
                    bw_value = float(seq_bw.replace('MB/s', '').strip())
                    if bw_value > 500:  # Over 500 MB/s is good
                        base_score = min(1.0, base_score + 0.1)
            except (ValueError, AttributeError):
                pass

        # Boost score for high random write performance
        if rand_bw != 'unknown' and isinstance(rand_bw, str):
            try:
                if 'MB/s' in rand_bw:
                    bw_value = float(rand_bw.replace('MB/s', '').strip())
                    if bw_value > 50:  # Over 50 MB/s for 4K random is excellent
                        base_score = min(1.0, base_score + 0.1)
            except (ValueError, AttributeError):
                pass

        return round(base_score, 2)

    def _get_default_devices(self):
        """
        Get default device configuration when no resource graph is available.
        
        Returns:
            List[Tuple[str, str, float]]: Default device configuration.
        """
        return [
            ('/tmp/cte_storage/cte_target.bin', '100GB', 0.6)
        ]

    def _validate_and_convert_devices(self, devices):
        """
        Validate and convert device specifications.
        
        Args:
            devices (List): Device specifications to validate.
            
        Returns:
            List[Tuple[str, str, float]]: Validated device tuples.
        """
        validated_devices = []
        
        for i, device in enumerate(devices):
            try:
                if isinstance(device, (list, tuple)) and len(device) >= 3:
                    path, capacity, score = device[0], device[1], device[2]
                    
                    # Validate path
                    if not isinstance(path, str) or not path.strip():
                        raise ValueError(f"Invalid path in device {i}: {path}")
                    
                    # Validate capacity using SizeType
                    capacity_str = self._normalize_capacity_with_sizetype(capacity)
                    
                    # Validate score
                    score_float = float(score)
                    if not 0.0 <= score_float <= 1.0:
                        raise ValueError(f"Score must be between 0.0 and 1.0, got: {score_float}")
                    
                    validated_devices.append((path.strip(), capacity_str, score_float))
                else:
                    raise ValueError(f"Device {i} must be a tuple/list with at least 3 elements")
                    
            except Exception as e:
                self.log(f"Warning: Invalid device specification {i}: {device} - {e}")
                continue
        
        if not validated_devices:
            self.log("Warning: No valid devices found, using defaults")
            return self._get_default_devices()
        
        return validated_devices

    def _normalize_capacity_with_sizetype(self, capacity):
        """
        Normalize capacity specification using Jarvis SizeType utility.
        
        Args:
            capacity: Capacity specification (string or number).
            
        Returns:
            str: Normalized capacity string.
        """
        try:
            # Convert capacity to SizeType - it handles the string formatting automatically
            size_type = SizeType(capacity)
            return str(size_type)
                
        except Exception as e:
            raise ValueError(f"Invalid capacity format '{capacity}': {e}")

    def _build_compose_config(self, devices):
        """
        Build a chimaera compose compatible configuration dictionary.

        Args:
            devices (List[Tuple]): List of device tuples (path, capacity, score).

        Returns:
            dict: Complete chimaera compose compatible configuration.
        """
        # Convert devices to storage configuration format
        storage_config = []
        for path, capacity, score in devices:
            # Determine bdev_type: check if path begins with ram:: prefix
            if path.startswith('ram::'):
                bdev_type = 'ram'
            else:
                bdev_type = 'file'

            storage_config.append({
                'path': path,
                'bdev_type': bdev_type,
                'capacity_limit': capacity,
                'score': score
            })

        # Build compose entry for wrp_cte_core module
        compose_entry = {
            'mod_name': 'wrp_cte_core',
            'pool_name': self.config.get('pool_name', 'wrp_cte_core'),
            'pool_query': self.config.get('pool_query', 'local'),
            'pool_id': self.config.get('pool_id', 512.0),
            'targets': {
                'neighborhood': self.config.get('neighborhood', 4),
                'default_target_timeout_ms': self.config.get('default_target_timeout_ms', 30000),
                'poll_period_ms': self.config.get('poll_period_ms', 5000)
            },
            'storage': storage_config,
            'dpe': {
                'dpe_type': self.config.get('dpe_type', 'max_bw')
            },
            'compression': {
                'monitor_interval_ms': self.config.get('monitor_interval_ms', 5),
                'dnn_model_weights_path': self.config.get('dnn_model_weights_path', ''),
                'dnn_samples_before_reinforce': self.config.get('dnn_samples_before_reinforce', 1000),
                'trace_folder_path': self.config.get('trace_folder_path', '')
            }
        }

        # Build compose list starting with core module
        compose_list = [compose_entry]

        # Conditionally add compressor module when compression is enabled
        iowarp_compress = self.config.get('iowarp_compress', 'none').lower()
        if iowarp_compress not in ['none', 'off', '']:
            compressor_entry = {
                'mod_name': 'wrp_cte_compressor',
                'pool_name': 'wrp_cte_compressor',
                'pool_query': self.config.get('pool_query', 'local'),
                'pool_id': self.config.get('pool_id', 512.0) + 1
            }
            compose_list.append(compressor_entry)
            self.log(f"Compression enabled ({iowarp_compress}), adding wrp_cte::compressor to compose")

        # Build complete compose configuration
        config = {
            'compose': compose_list
        }

        return config

    def start(self):
        """
        Start the WrpCte service by launching CTE using chimaera compose.

        This method executes the chimaera compose utility with PsshExecInfo
        to deploy the Content Transfer Engine across all nodes.

        Returns:
            bool: True if chimaera compose executed successfully, False otherwise.
        """
        self.log("Starting Content Transfer Engine using chimaera compose...")

        if not self.compose_config_path:
            self.log("Error: CTE not configured. Run configure first.")
            return False

        if not os.path.exists(self.compose_config_path):
            self.log(f"Error: Compose config file not found: {self.compose_config_path}")
            return False

        # Build the chimaera compose command
        cmd = f'chimaera compose {self.compose_config_path}'

        self.log(f"Running: {cmd}")
        self.log(f"  Config: {self.compose_config_path}")
        self.log(f"  Nodes: {len(self.jarvis.hostfile)}")

        # Prepare environment with compression settings for IOWarp Engine
        env = self.env.copy() if self.env else {}
        iowarp_compress = self.config.get('iowarp_compress', 'none')
        iowarp_compress_trace = self.config.get('iowarp_compress_trace', 'off')

        env['IOWARP_COMPRESS'] = iowarp_compress
        env['IOWARP_COMPRESS_TRACE'] = iowarp_compress_trace

        self.log(f"  Compression environment: IOWARP_COMPRESS={iowarp_compress}, IOWARP_COMPRESS_TRACE={iowarp_compress_trace}")

        try:
            # Execute chimaera compose on all nodes using PsshExecInfo
            Exec(cmd, PsshExecInfo(
                env=env,
                hostfile=self.jarvis.hostfile
            )).run()

            self.log("Content Transfer Engine started successfully on all nodes")
            return True
        except Exception as e:
            self.log(f"Error starting CTE: {e}")
            return False

    def stop(self):
        """
        Stop the WrpCte service.
        
        Since this is a configuration-only service, this method just passes.
        
        Returns:
            bool: Always True since this is configuration-only.
        """
        self.log("WrpCte is a configuration service - no persistent process to stop")
        return True

    def kill(self):
        """
        Force stop the WrpCte service.
        
        Since this is a configuration-only service, this method just passes.
        
        Returns:
            bool: Always True since this is configuration-only.
        """
        self.log("WrpCte is a configuration service - no persistent process to kill")
        return True

    def status(self):
        """
        Check the current status of the WrpCte service.

        Returns:
            dict: A dictionary containing service status information.
        """
        if self.compose_config_path and os.path.exists(self.compose_config_path):
            return {
                "running": True,
                "details": "CTE compose configuration active",
                "config_file": self.compose_config_path
            }
        else:
            return {
                "running": False,
                "details": "CTE compose configuration not generated",
                "config_file": None
            }

    def clean(self):
        """
        Clean up CTE compose configuration files and storage devices using Rm with PsshExecInfo.
        """
        self.log("Starting CTE cleanup process...")

        # Clean up compose configuration file
        if self.compose_config_path and os.path.exists(self.compose_config_path):
            try:
                os.remove(self.compose_config_path)
                self.log(f"Removed CTE compose configuration file: {self.compose_config_path}")
            except Exception as e:
                self.log(f"Error removing compose configuration file: {e}")

        # Clean up storage devices using Rm with PsshExecInfo
        try:
            # Get all configured storage devices
            devices = self.config.get('devices', [])

            # If no devices were manually configured, get from resource graph
            if not devices:
                devices.extend(self._get_devices_from_resource_graph())

            # Clean up each device and parent directory using Rm with PsshExecInfo
            if devices:
                self.log(f"Cleaning up {len(devices)} storage devices...")

                for path, _, _ in devices:
                    # Skip RAM devices (they don't have files to clean)
                    if path.startswith('ram::'):
                        self.log(f"Skipping RAM device: {path}")
                        continue

                    try:
                        # Execute removal using Rm with PsshExecInfo across all nodes
                        Rm(path, PsshExecInfo(hostfile=self.hostfile)).run()
                        self.log(f"Successfully cleaned storage device: {path}")

                        # Remove parent directory
                        parent_dir = os.path.dirname(path)
                        if parent_dir:
                            self.log(f"Removing directory: {parent_dir}")
                            Rm(parent_dir, PsshExecInfo(hostfile=self.hostfile)).run()
                            self.log(f"Successfully removed parent directory: {parent_dir}")

                    except Exception as e:
                        self.log(f"Error cleaning storage device {path}: {e}")
            else:
                self.log("No storage devices to clean")

        except Exception as e:
            self.log(f"Error during storage device cleanup: {e}")

        self.log("CTE configuration cleanup completed")
