#!/usr/bin/env python3
"""
Mofka Consumer Benchmark
Consumes N events from a topic and measures throughput.

Usage:
    source env.sh
    python3 consumer.py [options]
"""
import argparse
import os
import sys
import time

def main():
    parser = argparse.ArgumentParser(description="Mofka Consumer Benchmark")
    parser.add_argument("--group-file", default=os.environ.get("MOFKA_GROUP_FILE", "/tmp/mofka-bench/mofka.json"))
    parser.add_argument("--topic", default=os.environ.get("MOFKA_TOPIC", "benchmark_topic"))
    parser.add_argument("--num-events", type=int, default=int(os.environ.get("MOFKA_NUM_EVENTS", "1000")))
    parser.add_argument("--data-size", type=int, default=int(os.environ.get("MOFKA_DATA_SIZE", "1024")))
    parser.add_argument("--num-threads", type=int, default=int(os.environ.get("MOFKA_NUM_THREADS", "1")))
    parser.add_argument("--data-selectivity", type=float, default=float(os.environ.get("MOFKA_DATA_SELECTIVITY", "1.0")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("MOFKA_BATCH_SIZE", "16")))
    parser.add_argument("--use-progress-thread", action="store_true", default=True)
    args = parser.parse_args()

    print("=" * 50)
    print("  Mofka Consumer Benchmark (Python)")
    print("=" * 50)
    print(f"  Group file:        {args.group_file}")
    print(f"  Topic:             {args.topic}")
    print(f"  Events to consume: {args.num_events}")
    print(f"  Data size:         {args.data_size} bytes")
    print(f"  Data selectivity:  {args.data_selectivity}")
    print(f"  Batch size:        {args.batch_size}")
    print(f"  Threads:           {args.num_threads}")
    print("=" * 50)

    if not os.path.exists(args.group_file):
        print(f"ERROR: Group file not found: {args.group_file}")
        print("  Did you start the server first? (./server.sh)")
        sys.exit(1)

    from mochi.mofka.client import MofkaDriver

    # Connect to Mofka
    driver = MofkaDriver(args.group_file, use_progress_thread=args.use_progress_thread)

    # Open topic
    topic = driver.open_topic(args.topic)

    # Create consumer without Python data_selector/data_broker callbacks.
    # Using Python callbacks with use_progress_thread=True causes a GIL
    # deadlock: the C++ progress thread tries to acquire the GIL for the
    # callback while the main thread holds it inside future.wait().
    # The default C++ consumer retrieves all event data without callbacks.
    consumer = topic.consumer(
        name="bench-consumer",
        batch_size=args.batch_size,
    )

    # Run benchmark
    # event.data returns a DataDescriptor (not raw bytes), so we track
    # transferred bytes as data_size * selectivity per event consumed.
    events_consumed = 0
    bytes_per_event = int(args.data_size * args.data_selectivity)

    print(f"\nConsuming {args.num_events} events...")
    t_start = time.time()

    while events_consumed < args.num_events:
        future = consumer.pull()
        event = future.wait()
        event.acknowledge()
        events_consumed += 1

    total_data_bytes = events_consumed * bytes_per_event

    t_end = time.time()

    elapsed_ms = (t_end - t_start) * 1000.0
    total_mb = total_data_bytes / (1024 * 1024)
    throughput_mbs = total_mb / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0
    events_per_sec = events_consumed / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0

    print(f"\n{'=' * 50}")
    print(f"  RESULTS")
    print(f"{'=' * 50}")
    print(f"  Events consumed:   {events_consumed}")
    print(f"  Total data:        {total_mb:.2f} MB")
    print(f"  Elapsed time:      {elapsed_ms:.2f} ms")
    print(f"  Throughput:        {throughput_mbs:.2f} MB/s")
    print(f"  Events/sec:        {events_per_sec:.0f}")
    print(f"{'=' * 50}")

    del consumer
    del topic
    del driver

if __name__ == "__main__":
    main()
