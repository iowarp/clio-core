Create a function called RouteTask in worker.cc. I want you to take the code from the Run function where it was calling ResolvePoolQuery and put it in here. This function  should detect if this is a local schedule and call the scheduling monitor functions (e.g., kLocalSchedule and kGlobalSchedule) like the Run function did. We should remove the functions from worker.cc dedicated to this (e.g., CallMonitorForLocalSchedule)

Add a flag to the base task called TASK_ROUTED. This bit is set immediately after kLocalSchedule is called in RouteTask. This indicates the task should not undergo additional re-routing. This bit should be checked at the beginning of RouteTask. If the bit is true, then return true. Otherwise continue with the function.

@CLAUDE.md

# LocalSerialize

In hshm, we have the class context-transport-primitives/include/hermes_shm/data_structures/serialization/local_serialize.h

I want you to write some unit tests verifying that it works for hshm::priv::string and hshm::priv::vector in their respective unit tests.

In addition, I want you to write a separate unit test verifying that it works for just basic types like std::vector and std::string and int.
Place this under test/unit/data_structures/serialization/test_local_serialize.cc. Add to the cmakes.


@CLAUDE.md

# LocalTaskArchive

context-runtime/include/chimaera/task_archives.h provides a serialization using cereal. 

We want to have something similar, but for local. We should create a new set of classes analagous to those.
Also make a new file called: context-runtime/include/chimaera/local_task_archives.h.
This will use hshm::LocalSerialize instead of cereal.

For local, bulk is handled differently. If the object is a ShmPtr, just serialize the ShmPtr value.
If the object is a FullPtr, just serialize the shm_ part of it. If it is a raw pointer, just
serialize the data as a full memory copy of the data.

Write a unit test to verify that the new methods created can correctly serialize and deserialize tasks.

@CLUADE.md

# Container & chi_refresh_repo

We will need to update Container to include methods for
serializing the task output using LocalSerialize.

We should add the methods:
1. LocalLoadIn
2. LocalSaveOut

These are effectively the same as their counterparts LoadIn and SaveOut.
Update chi_refresh_repo to do this. 
Use chi_refresh_repo on '/workspace/context-runtime/modules' , '/workspace/context-assimilation-engine', and '/workspace/context-transfer-engine afterwards.
Then ensure things compile.
If things fail to compile, then fix chi_refresh_repo and rerun.

@CLUADE.md

# Task futures

Async* operations will need to return a ``Future<Task>`` object instead of Task*. Future is a new template class you should create.

Future will store:
1. A FullPtr pointer to the Task
2. A FullPtr to a FutreShm object, which contains a hipc::vector representing the serialized task and an atomic is_complete_ bool. We should remove is_complete_ in the task as well.

Two constructors:
1. With AllocT* as input. It will allocate the FutureShm object. The FutureShm should inherit from ShmContainer.
2. With AllocT* and ShmPtr<FutureShm> as input.

@CLAUDE.md

ipc_manager currently has a function called Enqueue to place a task in the worker queues from clients or locally.
I want to change this design paradigm to be a little more flexible.
Instead, we should implement Send and Recv.
Here is how this change will need to be applied.

# IpcManager Send & Recv

We will need to replace ipc_manager->Enqueue with Send / Recv.
Remove Enqueue entirely from the IpcManager. 
Replace every instance of Enqueue with Send.

## Worker Queues

Update the worker_queue to store Future<TaskT> instead of ShmPtr<TaskT>.

## Send(FullPtr<TaskT> task)

1. Create Future on the stack.
2. Serialize the TaskT using a LocalTaskInArchive object. Let's use a std::vector for the serialization buffer. Reserve 4KB for the serialization buffer.
3. Copy the std::vector into the FutureShm's hipc::vector.
4. Enqueue the Future in the worker queue.
5. Return: Future<TaskT>

## Recv(const Future<Task> &task)

1. Poll for the completion of the atomic is_complete bool in FutureShm 
2. Deserialize the TaskT using LoadTaskOutArchive into Future's raw pointer.
3. Return nothing

## Chimods using futures for async

Move task->Wait to Future class. Code should be able to do Future->Wait() instead of task->Wait.
Update EVERY chimod to return Future<TaskT> from the Async* methods instead of a FullPtr<TaskT>. 

Update NewTask in IpcManager to use standard new instead of main_alloc_.
Update DelTask in IpcManager to use standard delete instead of Allocator::DelObj.
Update EVERY task to no longer take in ``CHI_MAIN_ALLOC_T *alloc`` as an input. For all tasks depending on it, please use HSHM_MALLOC instead.
Update EVERY *_runtime.cc code to take as input a Future<TaskT> instead of FullPtr<TaskT>.
Update the SendIn, SaveIn, LoadIn, LoadOut, LocalLoadIn, and LocalSaveOut methods to use take as input Future<TaskT> instead of FullPtr<TaskT> by updating chi_refresh_repo.

Comment out the admin SendIn, LoadIn, SendOut, and LoadOut method bodies. We will come back to those.

# Worker 

## Run
1. Pop will pop a future from the stack.
2. Set the FullPtr<FutureShm<Task>> to CHI_IPC->ToFullPtr(future_ptr.shm_). 
3. Call container->LocalLoadIn. This method should use NewTask to allocate the task first.
4. We will need to update several methods to take as input a Future instead of Task* in the worker class. 

## EndTask
1. Use container->LocalSaveOut to serialize task outputs into the hipc::vector in the future.
2. Call Future->Complete(). 

@CLAUDE.md

Add a new method to chi_refresh_repo called NewTask. 
NewTask will be a switch-case that does the following: 
``auto new_task_ptr = ipc_manager->NewTask<TASK_NAME>(); return new_task_ptr.template Cast<Task>();``
It should return a ``FullPtr<Task>``.
Call chi_refresh_repo on each chimod and ensure everything still compiles afterwards.

@CLAUDE.md
Update ProcessNewTasks, EndTask, and FutureShm.
FutureShm should also container the method_id from the task, not just the PoolId.

## ProcessNewTasks
Call container->NewTask to create a task based on the method_id, rather than NewTask directly.
Construct a ``Future<Task>`` object from the FullPtr<FutureShm> and the FullPtr<Task>. It should have a constructor
for this if it does not.
RunContext should store ``Future<Task>`` instead of FutureShm. 

## EndTask
EndTask should do:
1. container->LocalSaveOut(run_ctx->future_);
2. run_ctx->future_.SetComplete();
3. container->DelTask(run_ctx->future_.task_);

@CLAUDE.md

Let's divide the Future class into two classes Future and Promise.
Future should have the constructor ``Future(AllocT* alloc, hipc::FullPtr<TaskT> task_ptr)``.
Future should expose IsComplete() and Wait().
Promise should have the constructor ``Future(hipc::FullPtr<FutureT> future_shm, hipc::FullPtr<TaskT> task_ptr)``.
Promise should expose SetComplete().
RunContext should store Promise instead of Future.
We should update chi_refresh_repo to do Promise instead of Future for all inputs.

Let's add a new hipc::mpsc_queue to the WorkOrchestrator.
This queue should be called network_queue. 


@CLAUDE.md

# Task
Add a new flag called TASK_FIRE_AND_FORGET.
Add the SetFireAndForget, IsFireAndForget, and UnsetFireAndForget methods.

# Worker::EndTask
If the task is marked as TASK_FIRE_AND_FORGET, then delete the task.
It should check ``run_ctx->destroy_in_end_task_ || task->flags_.Any(TASK_FIRE_AND_FORGET)``
when deciding if to delete the task.
TASK_FIRE_AND_FORGET should only be checked in the non-remote part of the method.

# Admin::SendTask
Mark this task as TASK_FIRE_AND_FORGET.
Both SendIn and SendOut will never be awaited.

