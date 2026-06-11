#!/usr/bin/env python3
"""
Mofka Producer Benchmark
Sends N events to a topic and measures throughput.

Usage:
    source env.sh
    python3 producer.py [options]
"""
import argparse
import os
import sys
import time
import json

def main():
    parser = argparse.ArgumentParser(description="Mofka Producer Benchmark")
    parser.add_argument("--group-file", default=os.environ.get("MOFKA_GROUP_FILE", "/tmp/mofka-bench/mofka.json"))
    parser.add_argument("--topic", default=os.environ.get("MOFKA_TOPIC", "benchmark_topic"))
    parser.add_argument("--num-events", type=int, default=int(os.environ.get("MOFKA_NUM_EVENTS", "1000")))
    parser.add_argument("--data-size", type=int, default=int(os.environ.get("MOFKA_DATA_SIZE", "1024")))
    parser.add_argument("--metadata-size", type=int, default=int(os.environ.get("MOFKA_METADATA_SIZE", "64")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("MOFKA_BATCH_SIZE", "16")))
    parser.add_argument("--num-threads", type=int, default=int(os.environ.get("MOFKA_NUM_THREADS", "1")))
    parser.add_argument("--use-progress-thread", action="store_true", default=True)
    args = parser.parse_args()

    print("=" * 50)
    print("  Mofka Producer Benchmark (Python)")
    print("=" * 50)
    print(f"  Group file:     {args.group_file}")
    print(f"  Topic:          {args.topic}")
    print(f"  Events:         {args.num_events}")
    print(f"  Data size:      {args.data_size} bytes")
    print(f"  Metadata size:  {args.metadata_size} bytes")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Threads:        {args.num_threads}")
    print("=" * 50)

    if not os.path.exists(args.group_file):
        print(f"ERROR: Group file not found: {args.group_file}")
        print("  Did you start the server first? (./server.sh)")
        sys.exit(1)

    from mochi.mofka.client import MofkaDriver

    # Connect to Mofka.
    # use_progress_thread=False is a deliberate workaround for a producer
    # SIGSEGV (exit 139) at small payloads: with the progress thread on,
    # the C++ mofka::Data destructor calls _Py_Dealloc on the wrapped
    # Python bytes payload from an Argobots worker thread *without holding
    # the GIL*, racing the main thread's push loop. Moving RPC progress to
    # the main thread (which already holds the GIL) eliminates the race.
    # The upstream binding bug remains; see
    # pipelines/mofka/architecture_decisions.md.
    driver = MofkaDriver(args.group_file, use_progress_thread=False)

    # Open topic
    topic = driver.open_topic(args.topic)

    # Create producer with batch size
    # batch_size is an int parameter (not AdaptiveBatchSize class)
    producer = topic.producer(name="bench-producer", batch_size=args.batch_size)

    # Prepare data payload
    data_payload = b'\x42' * args.data_size

    # Prepare metadata template (padded to requested size)
    meta_base = {"id": 0}
    meta_str = json.dumps(meta_base)
    if len(meta_str) < args.metadata_size:
        meta_base["pad"] = "x" * max(0, args.metadata_size - len(meta_str) - 10)

    # Run benchmark
    print(f"\nProducing {args.num_events} events...")
    t_start = time.time()

    for i in range(args.num_events):
        meta_base["id"] = i
        metadata = json.dumps(meta_base)
        producer.push(metadata=metadata, data=data_payload)

    producer.flush()
    t_end = time.time()

    elapsed_ms = (t_end - t_start) * 1000.0
    total_bytes = args.num_events * (args.data_size + args.metadata_size)
    throughput_mbs = (total_bytes / (1024 * 1024)) / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0
    events_per_sec = args.num_events / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0

    print(f"\n{'=' * 50}")
    print(f"  RESULTS")
    print(f"{'=' * 50}")
    print(f"  Events produced:   {args.num_events}")
    print(f"  Total data:        {total_bytes / (1024*1024):.2f} MB")
    print(f"  Elapsed time:      {elapsed_ms:.2f} ms")
    print(f"  Throughput:        {throughput_mbs:.2f} MB/s")
    print(f"  Events/sec:        {events_per_sec:.0f}")
    print(f"{'=' * 50}")

    del producer
    del topic
    del driver

if __name__ == "__main__":
    main()
