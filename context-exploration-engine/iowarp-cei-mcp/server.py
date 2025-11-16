
"""
MCP server wrapping the Context Interface Python APIs.

This server exposes Context Interface operations through the Model Context Protocol:

Context Interface Operations (from test_bindings.py):
- put_blob: Store data in blobs under tags (context_bundle)
- list_blobs_in_tag: List blobs in a tag (context_query - list)
- get_blob_size: Get blob size (context_query - get size)
- get_blob: Retrieve blob data (context_query - get data)
- delete_blob: Delete blobs from tags (context_delete)

Additional CTE Operations:
- tag_query: Query tags by regex pattern
- blob_query: Query blobs by tag and blob regex patterns
- poll_telemetry_log: Poll telemetry log with time filter
- reorganize_blob: Reorganize blob placement with new score

Runtime Management:
- initialize_cte_runtime: Initialize CTE runtime (Chimaera runtime, client, and CTE subsystem)
- get_client_status: Get CTE client initialization status
- get_cte_types: Get available CTE types and operations

Note: This wraps the wrp_cte_core_ext Python bindings.
"""
import sys
import os
import json
import io
from pathlib import Path
from typing import Dict, List, Any, Optional
from contextlib import redirect_stderr, redirect_stdout

# Redirect stderr at module level for MCP communication
# This prevents C++ error messages from breaking JSON-RPC
_original_stderr = sys.stderr
_original_stdout = sys.stdout

from mcp.server.fastmcp import FastMCP

# Try to find and add CTE Python bindings to path
def _find_cte_bindings():
    """Try to locate CTE Python bindings and add to Python path."""
    # Common locations to search
    search_paths = [
        # Build directory (most likely location)
        Path(__file__).parent.parent.parent / "build" / "bin",
        Path(__file__).parent.parent.parent.parent / "build" / "bin",
        # System install locations
        Path("/usr/local/lib"),
        Path("/usr/lib"),
        # Current directory
        Path("."),
        # Environment variable
        os.environ.get("CTE_PYTHON_PATH", ""),
    ]
    
    # Also check if PYTHONPATH includes the build directory
    pythonpath = os.environ.get("PYTHONPATH", "")
    if pythonpath:
        for path in pythonpath.split(os.pathsep):
            if path:
                search_paths.append(Path(path))
    
    for search_path in search_paths:
        if not search_path:
            continue
        search_path = Path(search_path).resolve()
        
        if not search_path.exists():
            continue
        
        # Look for wrp_cte_core_ext.so or wrp_cte_core_ext*.so
        for pattern in ["wrp_cte_core_ext.so", "wrp_cte_core_ext*.so"]:
            matches = list(search_path.glob(pattern))
            if matches:
                # Add directory to Python path
                if str(search_path) not in sys.path:
                    sys.path.insert(0, str(search_path))
                return True
    
    return False

# Try to find bindings first
_found_bindings = _find_cte_bindings()

# Try to import CTE Python bindings
try:
    import wrp_cte_core_ext as cte
    CTE_AVAILABLE = True
except ImportError as e:
    if _found_bindings:
        print(f"Warning: Found bindings directory but still could not import: {e}", file=sys.stderr)
    else:
        print(f"Warning: Could not import wrp_cte_core_ext: {e}", file=sys.stderr)
        print("Note: CTE Python bindings must be built and available in Python path", file=sys.stderr)
        print(f"      Searched in: {[str(p) for p in _find_cte_bindings.__code__.co_consts if isinstance(p, str)][:5]}", file=sys.stderr)
        print(f"      Try: export PYTHONPATH=/workspace/build/bin:$PYTHONPATH", file=sys.stderr)
    CTE_AVAILABLE = False
    cte = None

mcp = FastMCP("IOWarp Context Transfer Engine (CTE) MCP Server")

# Global client and initialization state
_initialized = False
_runtime_initialized = False
_client = None
_mctx = None

def _initialize_runtime() -> bool:
    """Attempt to initialize CTE runtime.
    
    Returns True if initialization successful or already initialized, False otherwise.
    
    Note: Suppresses stdout/stderr during initialization to avoid interfering with JSON-RPC.
    """
    global _runtime_initialized
    
    if _runtime_initialized:
        return True
    
    if not CTE_AVAILABLE:
        return False
    
    # Suppress stderr during initialization using OS-level file descriptor redirection
    # This prevents C++ errors from breaking JSON-RPC
    # Following test_bindings.py initialization pattern
    import time
    
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        old_stderr_fd = os.dup(2)  # Save original stderr
        
        try:
            # Redirect stderr to /dev/null
            os.dup2(devnull_fd, 2)
            
            # Setup environment paths
            try:
                module_file = cte.__file__ if hasattr(cte, '__file__') else None
                if module_file:
                    bin_dir = os.path.dirname(os.path.abspath(module_file))
                    os.environ["CHI_REPO_PATH"] = bin_dir
                    existing_ld_path = os.getenv("LD_LIBRARY_PATH", "")
                    if existing_ld_path:
                        os.environ["LD_LIBRARY_PATH"] = f"{bin_dir}:{existing_ld_path}"
                    else:
                        os.environ["LD_LIBRARY_PATH"] = bin_dir
            except Exception:
                pass  # May fail, continue anyway
            
            # Get config path
            config_path = os.getenv("CHI_SERVER_CONF", "")
            
            # Step 1: Initialize Chimaera (unified init)
            chimaera_result = False
            if hasattr(cte, 'chimaera_init') and hasattr(cte, 'ChimaeraMode'):
                try:
                    chimaera_result = cte.chimaera_init(cte.ChimaeraMode.kClient, True)
                    if chimaera_result:
                        time.sleep(0.5)  # Give Chimaera time to initialize (500ms as per tests)
                except Exception:
                    pass  # May fail in some environments

            # Step 2: Initialize CTE subsystem (only if Chimaera succeeded)
            cte_result = False
            if chimaera_result and hasattr(cte, 'initialize_cte') and hasattr(cte, 'PoolQuery'):
                try:
                    pool_query = cte.PoolQuery.Dynamic()
                    cte_result = cte.initialize_cte(config_path, pool_query)
                except Exception:
                    pass  # May fail without proper config
        
        finally:
            # Restore original stderr
            os.dup2(old_stderr_fd, 2)
            os.close(old_stderr_fd)
            os.close(devnull_fd)
        
        # If CTE init succeeded, we're good
        if cte_result:
            _runtime_initialized = True
            return True
        
        # Mark as attempted so we don't keep trying
        _runtime_initialized = True
        return False
        
    except Exception as e:
        # Initialization failed - this is expected in some environments
        _runtime_initialized = True  # Mark as attempted
        return False

def _ensure_initialized() -> bool:
    """Ensure CTE is initialized. Returns True if ready, False otherwise.
    
    Note: This checks if the CTE client can be retrieved, but does NOT
    guarantee that CTE runtime is fully initialized. CTE runtime must be
    initialized separately before using query/reorganization functions.
    """
    global _initialized, _client, _mctx
    
    if not CTE_AVAILABLE:
        return False
    
    if _initialized:
        return True
    
    try:
        # Try to get the client - this will work if CTE is already initialized
        _client = cte.get_cte_client()
        _mctx = cte.MemContext()
        _initialized = True
        return True
    except Exception as e:
        # Client might not be initialized yet - that's okay for some operations
        # We'll handle this on a per-tool basis
        # Note: Even if we get the client, CTE runtime might not be initialized,
        # which will cause crashes when calling query functions. Those crashes
        # are caught in the individual tool functions.
        return False

def _get_pool_query_dynamic():
    """Get a PoolQuery::Dynamic() instance.
    
    Note: PoolQuery may not be bound in Python yet. This attempts to access it.
    """
    try:
        # Try to access PoolQuery if it's bound
        if hasattr(cte, 'PoolQuery'):
            return cte.PoolQuery.Dynamic()
        else:
            # If not bound, we can't perform queries that require it
            # Return None to indicate this limitation
            return None
    except Exception:
        return None

@mcp.tool()
def get_client_status() -> str:
    """Get the status of the CTE client connection and initialization."""
    if not CTE_AVAILABLE:
        return json.dumps({
            'available': False,
            'error': 'CTE Python bindings (wrp_cte_core_ext) not available',
            'message': 'CTE Python bindings must be built and available in Python path'
        }, indent=2)
    
    initialized = _ensure_initialized()
    pool_query_available = _get_pool_query_dynamic() is not None
    
    result = {
        'available': True,
        'initialized': initialized,
        'pool_query_available': pool_query_available,
        'message': 'CTE client is ready' if initialized else 'CTE client not yet initialized'
    }
    
    if not initialized:
        result['note'] = 'Some operations require CTE to be initialized first'
    if not pool_query_available:
        result['warning'] = 'PoolQuery not available - TagQuery and BlobQuery may not work'
    
    return json.dumps(result, indent=2)

@mcp.tool()
def tag_query(tag_regex: str, max_tags: int = 0) -> str:
    """Query tags by regex pattern. Returns a list of tag names matching the pattern.
    
    Args:
        tag_regex: Regular expression pattern to match tag names
        max_tags: Maximum number of tags to return (0 = unlimited)
    
    Returns:
        JSON with list of matching tag names
    """
    if not CTE_AVAILABLE:
        return json.dumps({
            'tag_regex': tag_regex,
            'max_tags': max_tags,
            'tags': [],
            'count': 0,
            'error': 'CTE Python bindings not available',
            'message': 'CTE Python bindings (wrp_cte_core_ext) must be built and available'
        }, indent=2)
    
    if not _ensure_initialized():
        return json.dumps({
            'error': 'CTE client not initialized. Initialize CTE first.'
        }, indent=2)
    
    # Try to initialize runtime if not already done
    if not _runtime_initialized:
        _initialize_runtime()
    
    pool_query = _get_pool_query_dynamic()
    if pool_query is None:
        return json.dumps({
            'error': 'PoolQuery not available in Python bindings',
            'note': 'TagQuery requires PoolQuery::Dynamic() which may not be bound yet'
        }, indent=2)
    
    try:
        # Attempt the query - this may fail if CTE runtime is not initialized
        tags = _client.TagQuery(_mctx, tag_regex, max_tags, pool_query)
        return json.dumps({
            'tag_regex': tag_regex,
            'max_tags': max_tags,
            'tags': list(tags),
            'count': len(tags)
        }, indent=2)
    except RuntimeError as e:
        return json.dumps({
            'tag_regex': tag_regex,
            'max_tags': max_tags,
            'tags': [],
            'count': 0,
            'error': 'CTE runtime error',
            'message': str(e),
            'note': 'CTE runtime may not be initialized. Initialize CTE runtime before using query functions.'
        }, indent=2)
    except Exception as e:
        # Catch any other exceptions (ValueError, AttributeError, etc.)
        return json.dumps({
            'tag_regex': tag_regex,
            'max_tags': max_tags,
            'tags': [],
            'count': 0,
            'error': str(type(e).__name__),
            'message': str(e),
            'note': 'Query failed - CTE runtime may not be initialized'
        }, indent=2)

@mcp.tool()
def blob_query(tag_regex: str, blob_regex: str, max_blobs: int = 0) -> str:
    """Query blobs by tag and blob regex patterns.
    
    Returns a list of (tag_name, blob_name) pairs matching the patterns.
    
    Args:
        tag_regex: Regular expression pattern to match tag names
        blob_regex: Regular expression pattern to match blob names
        max_blobs: Maximum number of blob results to return (0 = unlimited)
    
    Returns:
        JSON with list of matching (tag_name, blob_name) pairs
    """
    if not CTE_AVAILABLE:
        return json.dumps({
            'tag_regex': tag_regex,
            'blob_regex': blob_regex,
            'max_blobs': max_blobs,
            'blobs': [],
            'count': 0,
            'error': 'CTE Python bindings not available',
            'message': 'CTE Python bindings (wrp_cte_core_ext) must be built and available'
        }, indent=2)
    
    if not _ensure_initialized():
        return json.dumps({
            'error': 'CTE client not initialized. Initialize CTE first.'
        }, indent=2)
    
    # Try to initialize runtime if not already done
    if not _runtime_initialized:
        _initialize_runtime()
    
    pool_query = _get_pool_query_dynamic()
    if pool_query is None:
        return json.dumps({
            'error': 'PoolQuery not available in Python bindings',
            'note': 'BlobQuery requires PoolQuery::Dynamic() which may not be bound yet'
        }, indent=2)
    
    try:
        # Attempt the query - this may fail if CTE runtime is not initialized
        blobs = _client.BlobQuery(_mctx, tag_regex, blob_regex, max_blobs, pool_query)
        # Convert pairs to lists for JSON serialization
        blob_list = [(tag, blob) for tag, blob in blobs]
        return json.dumps({
            'tag_regex': tag_regex,
            'blob_regex': blob_regex,
            'max_blobs': max_blobs,
            'blobs': blob_list,
            'count': len(blob_list)
        }, indent=2)
    except RuntimeError as e:
        return json.dumps({
            'tag_regex': tag_regex,
            'blob_regex': blob_regex,
            'max_blobs': max_blobs,
            'blobs': [],
            'count': 0,
            'error': 'CTE runtime error',
            'message': str(e),
            'note': 'CTE runtime may not be initialized. Initialize CTE runtime before using query functions.'
        }, indent=2)
    except Exception as e:
        # Catch any other exceptions (ValueError, AttributeError, etc.)
        return json.dumps({
            'tag_regex': tag_regex,
            'blob_regex': blob_regex,
            'max_blobs': max_blobs,
            'blobs': [],
            'count': 0,
            'error': str(type(e).__name__),
            'message': str(e),
            'note': 'Query failed - CTE runtime may not be initialized'
        }, indent=2)

@mcp.tool()
def poll_telemetry_log(minimum_logical_time: int = 0) -> str:
    """Poll telemetry log with a minimum logical time filter.
    
    Args:
        minimum_logical_time: Minimum logical time for filtering entries (0 = all)
    
    Returns:
        JSON with list of telemetry entries
    """
    if not CTE_AVAILABLE:
        return json.dumps({
            'minimum_logical_time': minimum_logical_time,
            'entries': [],
            'count': 0,
            'error': 'CTE Python bindings not available',
            'message': 'CTE Python bindings (wrp_cte_core_ext) must be built and available'
        }, indent=2)
    
    if not _ensure_initialized():
        return json.dumps({
            'error': 'CTE client not initialized. Initialize CTE first.'
        }, indent=2)
    
    # Note: Auto-initialization disabled - call initialize_cte_runtime manually first
    
    try:
        # Attempt the query - this may fail if CTE runtime is not initialized
        telemetry = _client.PollTelemetryLog(_mctx, minimum_logical_time)
        
        # Serialize telemetry entries
        entries = []
        for entry in telemetry:
            entry_dict = {
                'op': str(entry.op_),
                'off': int(entry.off_),
                'size': int(entry.size_),
                'tag_id': {
                    'major': int(entry.tag_id_.major_),
                    'minor': int(entry.tag_id_.minor_),
                    'is_null': bool(entry.tag_id_.IsNull())
                },
                'logical_time': int(entry.logical_time_),
            }
            # Try to serialize timestamps if possible
            try:
                entry_dict['mod_time'] = str(entry.mod_time_)
                entry_dict['read_time'] = str(entry.read_time_)
            except Exception:
                pass
            entries.append(entry_dict)
        
        return json.dumps({
            'minimum_logical_time': minimum_logical_time,
            'entries': entries,
            'count': len(entries)
        }, indent=2)
    except RuntimeError as e:
        return json.dumps({
            'minimum_logical_time': minimum_logical_time,
            'entries': [],
            'count': 0,
            'error': 'CTE runtime error',
            'message': str(e),
            'note': 'CTE runtime may not be initialized. Initialize CTE runtime before using telemetry functions.'
        }, indent=2)
    except Exception as e:
        # Catch any other exceptions
        return json.dumps({
            'minimum_logical_time': minimum_logical_time,
            'entries': [],
            'count': 0,
            'error': str(type(e).__name__),
            'message': str(e),
            'note': 'Query failed - CTE runtime may not be initialized'
        }, indent=2)

@mcp.tool()
def reorganize_blob(tag_id_major: int, tag_id_minor: int, blob_name: str, new_score: float) -> str:
    """Reorganize blob placement with a new score for data placement optimization.
    
    Args:
        tag_id_major: Major component of the TagId
        tag_id_minor: Minor component of the TagId
        blob_name: Name of the blob to reorganize
        new_score: New score for blob placement (typically 0.0-1.0)
    
    Returns:
        JSON with reorganization result
    """
    if not CTE_AVAILABLE:
        return json.dumps({
            'tag_id': {
                'major': tag_id_major,
                'minor': tag_id_minor
            },
            'blob_name': blob_name,
            'new_score': new_score,
            'result_code': -1,
            'success': False,
            'error': 'CTE Python bindings not available',
            'message': 'CTE Python bindings (wrp_cte_core_ext) must be built and available'
        }, indent=2)
    
    if not _ensure_initialized():
        return json.dumps({
            'error': 'CTE client not initialized. Initialize CTE first.'
        }, indent=2)
    
    # Note: Auto-initialization disabled - call initialize_cte_runtime manually first
    
    try:
        # Create TagId
        tag_id = cte.TagId()
        tag_id.major_ = tag_id_major
        tag_id.minor_ = tag_id_minor
        
        # Attempt reorganization - this may fail if CTE runtime is not initialized
        result_code = _client.ReorganizeBlob(_mctx, tag_id, blob_name, new_score)
        
        return json.dumps({
            'tag_id': {
                'major': tag_id_major,
                'minor': tag_id_minor
            },
            'blob_name': blob_name,
            'new_score': new_score,
            'result_code': int(result_code),
            'success': result_code == 0,
            'message': 'Reorganization successful' if result_code == 0 else f'Reorganization failed with code {result_code}'
        }, indent=2)
    except RuntimeError as e:
        return json.dumps({
            'tag_id': {
                'major': tag_id_major,
                'minor': tag_id_minor
            },
            'blob_name': blob_name,
            'new_score': new_score,
            'result_code': -1,
            'success': False,
            'error': 'CTE runtime error',
            'message': str(e),
            'note': 'CTE runtime may not be initialized. Initialize CTE runtime before using reorganization functions.'
        }, indent=2)
    except Exception as e:
        # Catch any other exceptions
        return json.dumps({
            'tag_id': {
                'major': tag_id_major,
                'minor': tag_id_minor
            },
            'blob_name': blob_name,
            'new_score': new_score,
            'result_code': -1,
            'success': False,
            'error': str(type(e).__name__),
            'message': str(e),
            'note': 'Reorganization failed - CTE runtime may not be initialized'
        }, indent=2)

@mcp.tool()
def initialize_cte_runtime() -> str:
    """Initialize the CTE runtime (Chimaera runtime, client, and CTE subsystem).
    
    This function follows the initialization pattern from test_bindings.py:
    1. Setup environment paths (CHI_REPO_PATH, LD_LIBRARY_PATH)
    2. Use CHI_SERVER_CONF if available, otherwise try empty config
    3. Initialize Chimaera (chimaera_init with kClient mode, True) - wait 500ms
    4. Initialize CTE subsystem (initialize_cte with config_path)
    
    Returns:
        JSON with initialization status and detailed results
    """
    if not CTE_AVAILABLE:
        return json.dumps({
            'success': False,
            'error': 'CTE Python bindings not available',
            'message': 'CTE Python bindings (wrp_cte_core_ext) must be built and available'
        }, indent=2)
    
    global _runtime_initialized
    
    # Reset flag to allow retry
    was_initialized = _runtime_initialized
    _runtime_initialized = False
    
    import time
    import tempfile
    
    result = {
        'success': False,
        'runtime_init': False,
        'client_init': False,
        'cte_init': False,
        'messages': []
    }
    
    # Create a temporary file to log initialization progress
    log_file_path = os.path.join(tempfile.gettempdir(), f"cte_init_log_{os.getpid()}.json")
    
    def log_progress(data):
        with open(log_file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    log_progress(result) # Log initial state
    
    try:
        # Redirect file descriptors at OS level to suppress C++ stderr output
        # This prevents errors from breaking JSON-RPC communication
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        old_stderr_fd = os.dup(2)  # Save original stderr fd
        old_stdout_fd = os.dup(1)  # Save original stdout fd
        
        try:
            # Redirect stderr to /dev/null
            os.dup2(devnull_fd, 2)
            # Redirect stdout to /dev/null
            os.dup2(devnull_fd, 1)
            
            # Step 0: Setup environment paths (following test_bindings.py pattern)
            try:
                module_file = cte.__file__ if hasattr(cte, '__file__') else None
                if module_file:
                    bin_dir = os.path.dirname(os.path.abspath(module_file))
                    os.environ["CHI_REPO_PATH"] = bin_dir
                    existing_ld_path = os.getenv("LD_LIBRARY_PATH", "")
                    if existing_ld_path:
                        os.environ["LD_LIBRARY_PATH"] = f"{bin_dir}:{existing_ld_path}"
                    else:
                        os.environ["LD_LIBRARY_PATH"] = bin_dir
                    result['messages'].append(f'Set CHI_REPO_PATH={bin_dir}')
            except Exception as e:
                result['messages'].append(f'Could not set environment paths: {str(e)}')
            log_progress(result) # Log after setting env paths
            
            # Step 0.5: Get or generate config path
            # If CHI_SERVER_CONF is set, use it; otherwise try to generate a minimal config
            config_path = os.getenv("CHI_SERVER_CONF", "")
            
            if not config_path:
                # Try to generate a minimal config file (following test_bindings.py pattern)
                try:
                    import socket
                    import yaml
                    
                    def find_available_port(start_port=9129, end_port=9200):
                        """Find an available port in the given range"""
                        for port in range(start_port, end_port):
                            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                                try:
                                    s.bind(('', port))
                                    return port
                                except OSError:
                                    continue
                        return None
                    
                    # Find available port
                    port = find_available_port()
                    if port:
                        temp_dir = tempfile.gettempdir()
                        
                        # Create hostfile
                        hostfile = os.path.join(temp_dir, f"cte_mcp_hostfile_{os.getpid()}")
                        with open(hostfile, 'w') as f:
                            f.write("127.0.0.1\n")
                        
                        # Create storage directory
                        storage_dir = os.path.join(temp_dir, f"cte_mcp_storage_{os.getpid()}")
                        os.makedirs(storage_dir, exist_ok=True)
                        
                        # Generate config
                        config = {
                            'networking': {
                                'protocol': 'zmq',
                                'hostfile': hostfile,
                                'port': port
                            },
                            'workers': {
                                'num_workers': 2
                            },
                            'memory': {
                                'main_segment_size': '512M',
                                'client_data_segment_size': '256M',
                                'runtime_data_segment_size': '256M'
                            },
                            'devices': [
                                {
                                    'mount_point': storage_dir,
                                    'capacity': '512M'
                                }
                            ]
                        }
                        
                        # Write config
                        config_path = os.path.join(temp_dir, f"cte_mcp_conf_{os.getpid()}.yaml")
                        with open(config_path, 'w') as f:
                            yaml.dump(config, f)
                        
                        os.environ['CHI_SERVER_CONF'] = config_path
                        result['messages'].append(f'Generated config file: {config_path} (port: {port})')
                    else:
                        result['messages'].append('Could not find available port for config generation')
                except ImportError:
                    result['messages'].append('PyYAML not available - cannot generate config (install pyyaml or set CHI_SERVER_CONF)')
                except Exception as e:
                    result['messages'].append(f'Config generation failed: {type(e).__name__}: {str(e)}')
            else:
                result['messages'].append(f'Using config from CHI_SERVER_CONF: {config_path}')
            
            log_progress(result) # Log after getting/generating config path
            
            # Step 1: Initialize Chimaera (unified init - following test_bindings.py pattern)
            # Note: This may fail if runtime is already running or config is missing
            if hasattr(cte, 'chimaera_init') and hasattr(cte, 'ChimaeraMode'):
                try:
                    # Try to initialize - if it fails, it may return False or raise an exception
                    # In some cases, C++ FATAL may cause process abort which we can't catch
                    chimaera_result = cte.chimaera_init(cte.ChimaeraMode.kClient, True)
                    result['runtime_init'] = bool(chimaera_result)
                    result['client_init'] = bool(chimaera_result)
                    if chimaera_result:
                        # Give Chimaera time to initialize all components (500ms as per tests)
                        time.sleep(0.5)
                        result['messages'].append('Chimaera initialized successfully')
                    else:
                        # Init returned False - might already be initialized or failed silently
                        result['messages'].append('Chimaera init returned False - may already be initialized or needs external setup')
                        # Don't mark as failed yet - might still work
                        result['runtime_init'] = True  # Assume it's okay to proceed
                        result['client_init'] = True
                except SystemExit as e:
                    # Process may exit due to FATAL errors in C++ code - we can't prevent this
                    # But we try to return JSON before exit
                    result['messages'].append(f'Chimaera init caused process exit (code: {e.code}) - port may be in use or config invalid')
                    result['runtime_init'] = False
                    result['client_init'] = False
                    result['error'] = 'Process exit during Chimaera initialization'
                    # Try to return immediately before process exits
                    try:
                        os.dup2(old_stderr_fd, 2)
                        os.dup2(old_stdout_fd, 1)
                    except:
                        pass
                    return json.dumps(result, indent=2)
                except Exception as e:
                    result['messages'].append(f'Chimaera init failed: {type(e).__name__}: {str(e)}')
                    result['runtime_init'] = False
                    result['client_init'] = False
            else:
                result['messages'].append('chimaera_init not available in bindings')
            log_progress(result) # Log after Chimaera init

            # Step 2: Initialize CTE subsystem (following test_bindings.py pattern)
            if hasattr(cte, 'initialize_cte') and hasattr(cte, 'PoolQuery') and result['client_init']:
                try:
                    pool_query = cte.PoolQuery.Dynamic()
                    cte_result = cte.initialize_cte(config_path, pool_query)
                    result['cte_init'] = cte_result
                    if cte_result:
                        result['messages'].append('CTE subsystem initialized successfully')
                        result['success'] = True
                        _runtime_initialized = True
                        
                        # Step 4: Register storage target (required for PutBlob operations)
                        # Following test_bindings.py pattern
                        try:
                            client = cte.get_cte_client()
                            mctx = cte.MemContext()
                            
                            # Get storage directory from config if available
                            storage_dir = None
                            if config_path:
                                try:
                                    import yaml
                                    with open(config_path, 'r') as f:
                                        config = yaml.safe_load(f)
                                    devices = config.get('devices', [])
                                    if devices and len(devices) > 0:
                                        storage_dir = devices[0].get('mount_point')
                                except Exception:
                                    pass
                            
                            # Use default if not in config
                            if not storage_dir:
                                storage_dir = os.path.join(tempfile.gettempdir(), f"cte_mcp_storage_{os.getpid()}")
                                os.makedirs(storage_dir, exist_ok=True)
                            
                            # Create target path
                            target_path = os.path.join(storage_dir, "mcp_target")
                            os.makedirs(os.path.dirname(target_path), exist_ok=True)
                            
                            # Register file-based target (512MB size) with high pool ID to avoid conflicts
                            if hasattr(cte, 'BdevType') and hasattr(cte, 'PoolId') and hasattr(client, 'RegisterTarget'):
                                bdev_id = cte.PoolId(700, 0)
                                target_query = cte.PoolQuery.Local()
                                target_size = 512 * 1024 * 1024  # 512MB
                                
                                reg_result = client.RegisterTarget(mctx, target_path, cte.BdevType.kFile,
                                                                  target_size, target_query, bdev_id)
                                
                                if reg_result == 0:
                                    result['messages'].append(f'Storage target registered successfully: {target_path}')
                                    result['target_registered'] = True
                                    result['target_path'] = target_path
                                else:
                                    result['messages'].append(f'Storage target registration returned {reg_result} (may already be registered)')
                                    result['target_registered'] = reg_result == 0
                                    result['target_path'] = target_path
                            else:
                                result['messages'].append('RegisterTarget not available - PutBlob operations may fail')
                                result['target_registered'] = False
                        except Exception as e:
                            result['messages'].append(f'Could not register storage target: {type(e).__name__}: {str(e)}')
                            result['target_registered'] = False
                            result['messages'].append('PutBlob operations may fail without registered targets')
                    else:
                        result['messages'].append('CTE init returned False (may need proper config or external runtime)')
                except Exception as e:
                    result['messages'].append(f'CTE init failed: {str(e)}')
            elif not result['client_init']:
                result['messages'].append('Skipping CTE init (client not initialized)')
            else:
                result['messages'].append('initialize_cte or PoolQuery not available in bindings')
            log_progress(result) # Log after CTE init
        
        finally:
            # Always restore stderr and stdout before returning JSON
            # This is critical to ensure JSON-RPC communication works
            try:
                os.dup2(old_stderr_fd, 2)
                os.dup2(old_stdout_fd, 1)
                os.close(old_stderr_fd)
                os.close(old_stdout_fd)
                os.close(devnull_fd)
            except Exception:
                pass  # If restore fails, continue anyway
            
            # Flush stdout/stderr to ensure all output is written
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
        
        # Ensure client is initialized for queries
        if result['success']:
            _ensure_initialized()
        else:
            # Still mark as attempted to avoid repeated failures
            if not was_initialized:
                _runtime_initialized = True
        
        # Add helpful message if initialization failed
        if not result['success']:
            result['note'] = 'CTE runtime initialization may require external setup. Options: 1) Set CHI_SERVER_CONF to a valid config file path, 2) Ensure PyYAML is installed for automatic config generation (pip install pyyaml), 3) Ensure Chimaera runtime is not already running on the same port, 4) Use external Chimaera runtime setup. Note: If initialization fails with process exit, the C++ code may have called FATAL - check logs or try external setup.'
        log_progress(result) # Log before finally block
        
    except Exception as e:
        # Ensure we always return valid JSON, even on error
        result['error'] = str(e)
        result['messages'].append(f'Initialization error: {str(e)}')
        _runtime_initialized = True  # Mark as attempted
        log_progress(result) # Log in except block
    
    # Always return JSON (this prevents connection closing errors)
    try:
        return json.dumps(result, indent=2)
    except Exception:
        # Fallback if JSON serialization fails
        return json.dumps({
            'success': False,
            'error': 'Failed to serialize initialization result',
            'messages': ['Initialization attempted but failed']
        }, indent=2)

@mcp.tool()
def put_blob(tag_name: str, blob_name: str, data: str, offset: int = 0) -> str:
    """Create or get a tag and store blob data (context_bundle operation).
    
    This wraps the Context Interface context_bundle operation:
    - Creates or gets a tag by name
    - Stores blob data under the tag
    
    Args:
        tag_name: Name of the tag (created if doesn't exist)
        blob_name: Name of the blob to store
        data: Blob data as a string (will be converted to bytes)
        offset: Offset within blob to write data (default: 0)
    
    Returns:
        JSON with operation result
    """
    if not CTE_AVAILABLE:
        return json.dumps({
            'tag_name': tag_name,
            'blob_name': blob_name,
            'offset': offset,
            'success': False,
            'error': 'CTE Python bindings not available',
            'message': 'CTE Python bindings (wrp_cte_core_ext) must be built and available'
        }, indent=2)
    
    if not _ensure_initialized():
        return json.dumps({
            'error': 'CTE client not initialized. Initialize CTE runtime first.'
        }, indent=2)
    
    try:
        # Check if Tag class is available
        if not hasattr(cte, 'Tag'):
            return json.dumps({
                'tag_name': tag_name,
                'blob_name': blob_name,
                'offset': offset,
                'success': False,
                'error': 'Tag class not available',
                'message': 'Tag wrapper class is not bound in Python bindings'
            }, indent=2)
        
        # Create or get tag
        tag = cte.Tag(tag_name)
        
        # Convert string data to bytes
        blob_data = data.encode('utf-8') if isinstance(data, str) else data
        
        # Put blob data
        tag.PutBlob(blob_name, blob_data, offset)
        
        return json.dumps({
            'tag_name': tag_name,
            'blob_name': blob_name,
            'data_size': len(blob_data),
            'offset': offset,
            'success': True,
            'message': f'Successfully stored blob "{blob_name}" in tag "{tag_name}"'
        }, indent=2)
    except Exception as e:
        return json.dumps({
            'tag_name': tag_name,
            'blob_name': blob_name,
            'offset': offset,
            'success': False,
            'error': str(type(e).__name__),
            'message': str(e),
            'note': 'PutBlob may fail if runtime is not initialized or storage target is not registered'
        }, indent=2)

@mcp.tool()
def list_blobs_in_tag(tag_name: str) -> str:
    """List all blob names contained in a tag (context_query - list operation).
    
    This wraps the Context Interface context_query operation for listing blobs.
    
    Args:
        tag_name: Name of the tag to query
    
    Returns:
        JSON with list of blob names
    """
    if not CTE_AVAILABLE:
        return json.dumps({
            'tag_name': tag_name,
            'blobs': [],
            'count': 0,
            'error': 'CTE Python bindings not available'
        }, indent=2)
    
    if not _ensure_initialized():
        return json.dumps({
            'error': 'CTE client not initialized. Initialize CTE runtime first.'
        }, indent=2)
    
    try:
        # Check if Tag class is available
        if not hasattr(cte, 'Tag'):
            return json.dumps({
                'tag_name': tag_name,
                'blobs': [],
                'count': 0,
                'error': 'Tag class not available'
            }, indent=2)
        
        # Get tag
        tag = cte.Tag(tag_name)
        
        # Get list of blobs
        blob_list = tag.GetContainedBlobs()
        
        # Convert to list of strings for JSON serialization
        blobs = list(blob_list) if blob_list else []
        
        return json.dumps({
            'tag_name': tag_name,
            'blobs': blobs,
            'count': len(blobs)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            'tag_name': tag_name,
            'blobs': [],
            'count': 0,
            'error': str(type(e).__name__),
            'message': str(e)
        }, indent=2)

@mcp.tool()
def get_blob_size(tag_name: str, blob_name: str) -> str:
    """Get the size of a blob in a tag (context_query - get size operation).
    
    This wraps the Context Interface context_query operation for getting blob size.
    
    Args:
        tag_name: Name of the tag
        blob_name: Name of the blob
    
    Returns:
        JSON with blob size information
    """
    if not CTE_AVAILABLE:
        return json.dumps({
            'tag_name': tag_name,
            'blob_name': blob_name,
            'size': 0,
            'error': 'CTE Python bindings not available'
        }, indent=2)
    
    if not _ensure_initialized():
        return json.dumps({
            'error': 'CTE client not initialized. Initialize CTE runtime first.'
        }, indent=2)
    
    try:
        # Check if Tag class is available
        if not hasattr(cte, 'Tag'):
            return json.dumps({
                'tag_name': tag_name,
                'blob_name': blob_name,
                'size': 0,
                'error': 'Tag class not available'
            }, indent=2)
        
        # Get tag
        tag = cte.Tag(tag_name)
        
        # Get blob size
        blob_size = tag.GetBlobSize(blob_name)
        
        return json.dumps({
            'tag_name': tag_name,
            'blob_name': blob_name,
            'size': int(blob_size),
            'size_bytes': int(blob_size)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            'tag_name': tag_name,
            'blob_name': blob_name,
            'size': 0,
            'error': str(type(e).__name__),
            'message': str(e)
        }, indent=2)

@mcp.tool()
def get_blob(tag_name: str, blob_name: str, size: int = 0, offset: int = 0) -> str:
    """Retrieve blob data from a tag (context_query - get data operation).
    
    This wraps the Context Interface context_query operation for retrieving blob data.
    If size is 0, will use GetBlobSize to determine the size first.
    
    Args:
        tag_name: Name of the tag
        blob_name: Name of the blob to retrieve
        size: Size of data to retrieve (0 = use blob size)
        offset: Offset within blob to read from (default: 0)
    
    Returns:
        JSON with blob data (as base64 or string)
    """
    if not CTE_AVAILABLE:
        return json.dumps({
            'tag_name': tag_name,
            'blob_name': blob_name,
            'size': size,
            'offset': offset,
            'error': 'CTE Python bindings not available'
        }, indent=2)
    
    if not _ensure_initialized():
        return json.dumps({
            'error': 'CTE client not initialized. Initialize CTE runtime first.'
        }, indent=2)
    
    try:
        # Check if Tag class is available
        if not hasattr(cte, 'Tag'):
            return json.dumps({
                'tag_name': tag_name,
                'blob_name': blob_name,
                'error': 'Tag class not available'
            }, indent=2)
        
        # Get tag
        tag = cte.Tag(tag_name)
        
        # Get blob size if not specified
        if size == 0:
            blob_size = tag.GetBlobSize(blob_name)
            if blob_size == 0:
                return json.dumps({
                    'tag_name': tag_name,
                    'blob_name': blob_name,
                    'error': 'Blob not found or has zero size'
                }, indent=2)
            size = int(blob_size)
        
        # Get blob data
        blob_data = tag.GetBlob(blob_name, size, offset)
        
        # Convert to string if bytes
        if isinstance(blob_data, bytes):
            # Try to decode as UTF-8, fallback to base64 if binary
            try:
                data_str = blob_data.decode('utf-8')
                data_base64 = None
            except UnicodeDecodeError:
                import base64
                data_str = None
                data_base64 = base64.b64encode(blob_data).decode('utf-8')
        else:
            data_str = str(blob_data)
            data_base64 = None
        
        result = {
            'tag_name': tag_name,
            'blob_name': blob_name,
            'size': size,
            'offset': offset,
            'data_size': len(blob_data) if isinstance(blob_data, bytes) else len(str(blob_data))
        }
        
        if data_str:
            result['data'] = data_str
        if data_base64:
            result['data_base64'] = data_base64
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({
            'tag_name': tag_name,
            'blob_name': blob_name,
            'error': str(type(e).__name__),
            'message': str(e)
        }, indent=2)

@mcp.tool()
def delete_blob(tag_name: str, blob_name: str) -> str:
    """Delete a blob from a tag (context_delete operation).
    
    This wraps the Context Interface context_delete operation.
    
    Args:
        tag_name: Name of the tag
        blob_name: Name of the blob to delete
    
    Returns:
        JSON with deletion result
    """
    if not CTE_AVAILABLE:
        return json.dumps({
            'tag_name': tag_name,
            'blob_name': blob_name,
            'success': False,
            'error': 'CTE Python bindings not available'
        }, indent=2)
    
    if not _ensure_initialized():
        return json.dumps({
            'error': 'CTE client not initialized. Initialize CTE runtime first.'
        }, indent=2)
    
    try:
        # Check if Tag class and Client.DelBlob are available
        if not hasattr(cte, 'Tag'):
            return json.dumps({
                'tag_name': tag_name,
                'blob_name': blob_name,
                'success': False,
                'error': 'Tag class not available'
            }, indent=2)
        
        # Get tag to get TagId
        tag = cte.Tag(tag_name)
        tag_id = tag.GetTagId()
        
        # Delete blob using Client.DelBlob
        if not hasattr(_client, 'DelBlob'):
            return json.dumps({
                'tag_name': tag_name,
                'blob_name': blob_name,
                'success': False,
                'error': 'DelBlob method not available on Client'
            }, indent=2)
        
        result = _client.DelBlob(_mctx, tag_id, blob_name)
        
        return json.dumps({
            'tag_name': tag_name,
            'blob_name': blob_name,
            'success': bool(result),
            'message': f'Blob "{blob_name}" deleted from tag "{tag_name}"' if result else f'Failed to delete blob "{blob_name}"'
        }, indent=2)
    except Exception as e:
        return json.dumps({
            'tag_name': tag_name,
            'blob_name': blob_name,
            'success': False,
            'error': str(type(e).__name__),
            'message': str(e)
        }, indent=2)

@mcp.tool()
def get_cte_types() -> str:
    """Get information about available CTE types and operations.
    
    Returns:
        JSON with information about CTE types, enums, and available operations
    """
    result = {
        'available': CTE_AVAILABLE,
        'types': {},
        'operations': []
    }
    
    if not CTE_AVAILABLE:
        result['error'] = 'CTE Python bindings not available'
        result['message'] = 'CTE Python bindings (wrp_cte_core_ext) must be built and available'
        return json.dumps(result, indent=2)
    
    try:
        types_info = {
            'available': True,
            'types': {},
            'operations': []
        }
        
        # Get CteOp enum values
        if hasattr(cte, 'CteOp'):
            ops = []
            for attr in dir(cte.CteOp):
                if not attr.startswith('_'):
                    try:
                        value = getattr(cte.CteOp, attr)
                        ops.append(str(value))
                    except Exception:
                        pass
            types_info['operations'] = ops
        
        # Check for TagId/BlobId
        if hasattr(cte, 'TagId'):
            types_info['types']['TagId'] = 'Available'
        if hasattr(cte, 'BlobId'):
            types_info['types']['BlobId'] = 'Available'
        
        # Check for MemContext
        if hasattr(cte, 'MemContext'):
            types_info['types']['MemContext'] = 'Available'
        
        # Check for CteTelemetry
        if hasattr(cte, 'CteTelemetry'):
            types_info['types']['CteTelemetry'] = 'Available'
        
        # Check for PoolQuery (may not be bound)
        if hasattr(cte, 'PoolQuery'):
            types_info['types']['PoolQuery'] = 'Available'
        else:
            types_info['types']['PoolQuery'] = 'Not bound (required for queries)'
        
        return json.dumps(types_info, indent=2)
    except Exception as e:
        return json.dumps({
            'error': str(e)
        }, indent=2)

if __name__ == "__main__":
    mcp.run()
