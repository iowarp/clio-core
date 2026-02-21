# IOWarp Quickstart

Start a single-node IOWarp runtime with Docker in under a minute.

## Prerequisites

- Docker and Docker Compose installed
- At least 8 GB of available RAM

## Usage

```bash
# Start the runtime
docker compose up -d

# Check that the runtime started successfully
docker compose logs

# Follow logs in real time
docker compose logs -f

# Stop the runtime
docker compose down
```

## Configuration

Edit `chimaera.yaml` to customize the runtime. Key settings:

| Section | What it controls |
|---------|-----------------|
| `networking.port` | RPC listener port (default 9413) |
| `runtime.num_threads` | Worker threads (default 4) |
| `compose` | Modules loaded at startup |

See the [Configuration Reference](https://iowarp.readthedocs.io/en/latest/deployment/configuration/) for all options.

## Verifying

When the runtime starts successfully, you should see log output like:

```
SpawnWorkerThreads
```

This means the worker threads are running and the runtime is ready to accept connections.
