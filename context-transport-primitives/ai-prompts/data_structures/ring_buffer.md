@CLAUDE.md

# Ring Buffer

In the main branch, I have a ring_buffer implementation that provides various compile-time options, such as support for lock-free multiple-producer, single-consumer access. 

There are technically two, but I want you to ignore the ring_buffer_ptr_queue. Focus only on the ring_buffer.

I want you to adapt that to this current branch.

You should have hipc typedefs, but not hshm typedefs. Read the file to see what that means.

Instead of using hipc::pair for the queue, just make your own custom data structure for holding two entries.


@CLAUDE.md 

Please also add the relevant typedefs from the main branch. Every typedef from the ring_queue.h that is in hshm::ipc namespace please. Add them to the ring_buffer.h in this branch. These are the ones I remember:
1. ext_ring_buffer: An extensible ring buffer, single-thread only. It should extend buffer if we reach capacity limit.
2. spsc_ring_buffer: A fixed-size ring buffer, also single-thread only. It should error if we reach the capacity limit.
3. mpsc_ring_buffer: A fixed-size ring buffer, multiple can emplace, but only one can consume. It should NOT error if we reach capacity limit and assume the consumer will free up space eventually.

We should have a test verifying each typedef data structure.
We should have a single workload generator class testing all angles of the queues.
We may not use each workload for each typedef, but they should all be in a single class.
We should have a single source file for all ring buffer tests.
We have to have a SINGLE workload generator class for ALL ring_buffer queues. FOR ALL OF THEM. Not one for each, just a single class for ALL RING BUFFER QUEUES.
ONE SOURCE FILE!!! DO NOT MAKE SEPARATE SOURCE FILES FOR THE RING BUFFER TESTS!!! ONE FILE!!! ONE CLASS IN THE FILE FOR WORKLOAD GENERATION!!! AND THEN SEPARATE TESTS IN THAT FILE CALLING WORKLOAD GENERATOR FOR EACH QUEUE TYPE!!!! 

For mpsc_ring_buffer, we need the following test: 
1. We will spawn 4 producer threads. Each producer thread will emplace for 2 seconds. The queue should have capacity 8 to ensure there is contention among the threads.
2. We will spawn one consumer thread, which is polling the queue constantly. It will poll continuously for 4 seconds.
