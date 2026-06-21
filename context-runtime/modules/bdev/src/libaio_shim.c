/*
 * libaio_shim.c — implements the four libaio functions used by bdev via direct
 * Linux syscalls, so libclio_bdev_runtime.so has no dependency on libaio.so.
 *
 * io_prep_pread / io_prep_pwrite / io_set_eventfd are static inlines in
 * libaio.h and need no implementation here.
 */
#define _GNU_SOURCE
#include <unistd.h>
#include <sys/syscall.h>
#include <libaio.h>

int io_setup(unsigned nr_events, io_context_t *ctx)
{
    return (int)syscall(SYS_io_setup, nr_events, ctx);
}

int io_destroy(io_context_t ctx)
{
    return (int)syscall(SYS_io_destroy, ctx);
}

int io_submit(io_context_t ctx, long nr, struct iocb **iocbpp)
{
    return (int)syscall(SYS_io_submit, ctx, nr, iocbpp);
}

int io_getevents(io_context_t ctx, long min_nr, long nr,
                 struct io_event *events, struct timespec *timeout)
{
    return (int)syscall(SYS_io_getevents, ctx, min_nr, nr, events, timeout);
}
