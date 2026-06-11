#!/usr/bin/env python3
"""
Create the Mofka topic + partition via the MofkaDriver Python API.

This replaces the `mofkactl` CLI calls that the server used to make
(`mofkactl topic create` / `mofkactl partition add`). On current mochi
builds mofkactl is broken by a typer/Click version conflict — Click 8.2.0
changed `Parameter.make_metavar()` to require a `ctx` argument, and the
bundled typer calls it the old way, so any mofkactl invocation that renders
options crashes with:

    TypeError: Parameter.make_metavar() missing 1 required positional argument: 'ctx'

The `mochi.mofka.client` bindings themselves are fine, so we drive topic and
partition creation directly through MofkaDriver. Idempotent: a no-op if the
topic already exists. Prints a RESULTS line on success so the caller can
assert on it (same convention as producer.py / consumer.py).

Usage:
    python3 setup_topic.py --group-file <mofka.json> --topic <name> \
        [--partition-type memory] [--server-rank 0]
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Mofka topic/partition setup")
    parser.add_argument(
        "--group-file",
        default=os.environ.get("MOFKA_GROUP_FILE", "/tmp/mofka-bench/mofka.json"))
    parser.add_argument(
        "--topic", default=os.environ.get("MOFKA_TOPIC", "benchmark_topic"))
    parser.add_argument("--partition-type", default="memory")
    parser.add_argument("--server-rank", type=int, default=0)
    args = parser.parse_args()

    print("=" * 50)
    print("  Mofka Topic/Partition Setup (MofkaDriver API)")
    print("=" * 50)
    print(f"  Group file:     {args.group_file}")
    print(f"  Topic:          {args.topic}")
    print(f"  Partition type: {args.partition_type}")
    print(f"  Server rank:    {args.server_rank}")
    print("=" * 50)

    if not os.path.exists(args.group_file):
        print(f"ERROR: Group file not found: {args.group_file}")
        print("  Did bedrock start and write mofka.json?")
        sys.exit(1)

    if args.partition_type != "memory":
        # The benchmark only exercises in-memory partitions; other targets
        # (disk-backed warabi) would use add_custom_partition with a config.
        print(f"ERROR: unsupported partition type '{args.partition_type}' "
              f"(only 'memory' is wired up)")
        sys.exit(1)

    from mochi.mofka.client import MofkaDriver

    # Admin-only operations (no bytes payloads), so the producer's
    # use_progress_thread=False workaround is unnecessary here.
    driver = MofkaDriver(args.group_file)

    if driver.topic_exists(args.topic):
        print(f"Topic '{args.topic}' already exists — skipping create")
    else:
        driver.create_topic(args.topic)
        print(f"Created topic '{args.topic}'")
        driver.add_memory_partition(args.topic, args.server_rank)
        print(f"Added memory partition on rank {args.server_rank}")

    print("RESULTS: mofka topic+partition ready")

    del driver


if __name__ == "__main__":
    main()
