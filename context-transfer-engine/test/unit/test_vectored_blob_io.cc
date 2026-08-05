/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved. BSD 3-Clause license.
 */

// Vectored PutBlob / GetBlob (issue #820).
//
// A vectored task carries N regions instead of one, so the runtime acquires the
// blob's #680 write token ONCE and applies every region in a single pass. This
// is what lets the worker's batching layer merge N independent same-blob tasks
// without gathering their payloads into one contiguous buffer.
//
// What is asserted, in order of importance:
//   1. EQUIVALENCE -- a vectored put of N disjoint regions leaves the blob
//      byte-identical to N separate single-region puts.
//   2. ORDERING -- when two segments cover the same bytes, the LATER one in the
//      list wins. This is the guarantee N racing single-region tasks do NOT
//      provide (they race the write token), and it is why batching may merge
//      overlapping writes at all.
//   3. The union range is what gets allocated: a vectored put whose segments
//      start past EOF grows the blob to cover them, and the hole reads as zeros.
//   4. Vectored GetBlob scatters each region into its OWN buffer.

#include <clio_runtime/clio_runtime.h>
#include <clio_runtime/bdev/bdev_client.h>
#include <clio_cte/core/core_client.h>
#include <clio_cte/core/core_tasks.h>

#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "simple_test.h"

namespace {

const char *kTargetName = "vectored_blob_target";
constexpr clio::run::u64 kTargetSize = 1ULL * 1024 * 1024 * 1024;  // 1 GiB

class Fixture {
 public:
  bool initialized_ = false;
  Fixture() {
    bool ok = clio::run::CLIO_INIT(clio::run::RuntimeMode::kClient, true);
    REQUIRE(ok);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    ok = clio::cte::core::CLIO_CTE_CLIENT_INIT();
    REQUIRE(ok);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto *cte = CLIO_CTE_CLIENT;
    clio::run::PoolId bdev_pool_id(911, 0);
    clio::run::bdev::Client bdev_client(bdev_pool_id);
    auto create = bdev_client.AsyncCreate(
        clio::run::PoolQuery::Dynamic(), kTargetName, bdev_pool_id,
        clio::run::bdev::BdevType::kRam, kTargetSize);
    create.Wait();
    auto reg = cte->AsyncRegisterTarget(kTargetName,
                                        clio::run::bdev::BdevType::kRam,
                                        kTargetSize, clio::run::PoolQuery::Local(),
                                        bdev_pool_id);
    reg.Wait();
    REQUIRE(reg->GetReturnCode() == 0);
    initialized_ = true;
  }
};

Fixture *g_fixture = nullptr;

/** Allocate an SHM buffer of `n` bytes filled with `v`. */
ctp::ipc::FullPtr<char> FillBuf(size_t n, char v) {
  auto *ipc = CLIO_IPC;
  ctp::ipc::FullPtr<char> b = ipc->AllocateBuffer(n);
  REQUIRE(!b.IsNull());
  std::memset(b.ptr_, v, n);
  return b;
}

/** Read [off, off+n) of a blob through an ordinary single-region GetBlob. */
std::vector<char> ReadBack(clio::cte::core::TagId tag_id,
                           const std::string &blob, clio::run::u64 off,
                           size_t n) {
  auto *ipc = CLIO_IPC;
  auto *cte = CLIO_CTE_CLIENT;
  ctp::ipc::FullPtr<char> buf = ipc->AllocateBuffer(n);
  REQUIRE(!buf.IsNull());
  std::memset(buf.ptr_, 0, n);
  auto g = cte->AsyncGetBlob(tag_id, blob, off, n, /*flags=*/0,
                             ctp::ipc::ShmPtr<>(buf.shm_));
  g.Wait();
  REQUIRE(g->GetReturnCode() == 0);
  std::vector<char> out(buf.ptr_, buf.ptr_ + n);
  ipc->FreeBuffer(buf);
  return out;
}

}  // namespace

TEST_CASE("Vectored PutBlob == N single puts, and segments apply in list order",
          "[cte][vectored][820]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);
  auto *ipc = CLIO_IPC;
  auto *cte = CLIO_CTE_CLIENT;

  clio::cte::core::Tag tag("vectored_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();

  const size_t kSeg = 4096;
  const int kN = 64;  // 64 disjoint 4 KiB regions = 256 KiB

  // ---- 1. EQUIVALENCE ----------------------------------------------------
  // Write kN disjoint regions two ways and require identical bytes.
  {
    // (a) N separate single-region puts.
    const std::string blob_a = "single";
    for (int i = 0; i < kN; ++i) {
      auto b = FillBuf(kSeg, static_cast<char>((i % 250) + 1));
      auto p = cte->AsyncPutBlob(tag_id, blob_a,
                                 static_cast<clio::run::u64>(i) * kSeg, kSeg,
                                 ctp::ipc::ShmPtr<>(b.shm_));
      p.Wait();
      REQUIRE(p->GetReturnCode() == 0);
      ipc->FreeBuffer(b);
    }

    // (b) ONE vectored put carrying the same kN regions.
    const std::string blob_b = "vectored";
    std::vector<ctp::ipc::FullPtr<char>> bufs;
    std::vector<clio::cte::core::BlobSegment> segs;
    for (int i = 0; i < kN; ++i) {
      auto b = FillBuf(kSeg, static_cast<char>((i % 250) + 1));
      bufs.push_back(b);
      segs.push_back(clio::cte::core::BlobSegment(
          static_cast<clio::run::u64>(i) * kSeg, kSeg,
          ctp::ipc::ShmPtr<>(b.shm_)));
    }
    auto pv = cte->AsyncPutBlobVectored(tag_id, blob_b, segs);
    pv.Wait();
    REQUIRE(pv->GetReturnCode() == 0);
    for (auto &b : bufs) ipc->FreeBuffer(b);

    auto a = ReadBack(tag_id, blob_a, 0, kSeg * kN);
    auto v = ReadBack(tag_id, blob_b, 0, kSeg * kN);
    REQUIRE(a.size() == v.size());
    REQUIRE(std::memcmp(a.data(), v.data(), a.size()) == 0);
    // ...and the bytes are actually the pattern we wrote, not zeros on both.
    for (int i = 0; i < kN; ++i) {
      REQUIRE(v[static_cast<size_t>(i) * kSeg] ==
              static_cast<char>((i % 250) + 1));
    }
    std::printf("[#820] vectored put of %d x %zu B == %d single puts\n", kN,
                kSeg, kN);
  }

  // ---- 2. ORDERING: later segment wins on overlap ------------------------
  // Three segments all covering [0, kSeg). The LAST one in the list must be
  // the surviving value. N racing single-region tasks cannot promise this.
  {
    const std::string blob = "overlap";
    std::vector<ctp::ipc::FullPtr<char>> bufs;
    std::vector<clio::cte::core::BlobSegment> segs;
    for (char v : {'A', 'B', 'C'}) {
      auto b = FillBuf(kSeg, v);
      bufs.push_back(b);
      segs.push_back(clio::cte::core::BlobSegment(0, kSeg,
                                                  ctp::ipc::ShmPtr<>(b.shm_)));
    }
    auto pv = cte->AsyncPutBlobVectored(tag_id, blob, segs);
    pv.Wait();
    REQUIRE(pv->GetReturnCode() == 0);
    for (auto &b : bufs) ipc->FreeBuffer(b);

    auto got = ReadBack(tag_id, blob, 0, kSeg);
    for (size_t i = 0; i < kSeg; ++i) {
      REQUIRE(got[i] == 'C');
    }
    std::printf("[#820] overlapping segments resolve last-writer-wins\n");
  }

  // ---- 3. Union range is allocated; the gap reads as zeros ---------------
  // Two far-apart segments in one task: the blob must grow to cover the union
  // and the hole between them must read back as zeros (POSIX sparse-write).
  {
    const std::string blob = "sparse";
    auto b0 = FillBuf(kSeg, 'X');
    auto b1 = FillBuf(kSeg, 'Y');
    std::vector<clio::cte::core::BlobSegment> segs;
    segs.push_back(clio::cte::core::BlobSegment(0, kSeg,
                                                ctp::ipc::ShmPtr<>(b0.shm_)));
    const clio::run::u64 far = 64 * 1024;
    segs.push_back(clio::cte::core::BlobSegment(far, kSeg,
                                                ctp::ipc::ShmPtr<>(b1.shm_)));
    auto pv = cte->AsyncPutBlobVectored(tag_id, blob, segs);
    pv.Wait();
    REQUIRE(pv->GetReturnCode() == 0);
    ipc->FreeBuffer(b0);
    ipc->FreeBuffer(b1);

    auto got = ReadBack(tag_id, blob, 0, far + kSeg);
    for (size_t i = 0; i < kSeg; ++i) REQUIRE(got[i] == 'X');
    for (size_t i = kSeg; i < far; ++i) REQUIRE(got[i] == 0);
    for (size_t i = 0; i < kSeg; ++i) REQUIRE(got[far + i] == 'Y');
    std::printf("[#820] union range allocated; inter-segment hole reads zero\n");
  }

  // ---- 4. Vectored GET scatters into per-segment buffers -----------------
  // Lay down a known pattern, then read it back as N regions in ONE task, each
  // into its own buffer. Every buffer must hold exactly its own region — this
  // is what removes the scatter copy from the batching layer.
  {
    const std::string blob = "vecget";
    for (int i = 0; i < kN; ++i) {
      auto b = FillBuf(kSeg, static_cast<char>((i % 250) + 1));
      auto p = cte->AsyncPutBlob(tag_id, blob,
                                 static_cast<clio::run::u64>(i) * kSeg, kSeg,
                                 ctp::ipc::ShmPtr<>(b.shm_));
      p.Wait();
      REQUIRE(p->GetReturnCode() == 0);
      ipc->FreeBuffer(b);
    }

    std::vector<ctp::ipc::FullPtr<char>> dst;
    std::vector<clio::cte::core::BlobSegment> segs;
    for (int i = 0; i < kN; ++i) {
      auto b = FillBuf(kSeg, 0);
      dst.push_back(b);
      segs.push_back(clio::cte::core::BlobSegment(
          static_cast<clio::run::u64>(i) * kSeg, kSeg,
          ctp::ipc::ShmPtr<>(b.shm_)));
    }
    auto gv = cte->AsyncGetBlobVectored(tag_id, blob, segs);
    gv.Wait();
    REQUIRE(gv->GetReturnCode() == 0);

    for (int i = 0; i < kN; ++i) {
      const char want = static_cast<char>((i % 250) + 1);
      for (size_t j = 0; j < kSeg; ++j) {
        REQUIRE(dst[i].ptr_[j] == want);
      }
    }
    for (auto &b : dst) ipc->FreeBuffer(b);
    std::printf("[#820] vectored get scattered %d regions into own buffers\n",
                kN);
  }
}

// The per-segment validation branches in PutBlobImpl/GetBlobImpl: a vectored
// task with a zero-size or null-buffer segment must be REFUSED, not silently
// half-applied. These are cheap, deterministic, and were the runtime's untested
// error paths.
TEST_CASE("Vectored PRIVATE-memory put/get roundtrip (strided)",
          "[cte][vectored][private]") {
  // The PrivBlobSegment overloads: put N strided regions from caller-owned
  // char* buffers, read them back into different char* buffers, byte-compare.
  // Under the embedded runtime this exercises the FromRaw direct paths and,
  // on the second read, the all-or-nothing SHM fast path (RAM target).
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);
  auto *cte = CLIO_CTE_CLIENT;

  clio::cte::core::Tag tag("vectored_priv_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();
  const std::string blob = "priv_strided";

  const size_t kSeg = 4096;
  const int kN = 16;
  const clio::run::u64 kStride = 2 * kSeg;  // strided: gap between regions

  std::vector<std::vector<char>> src(kN), dst(kN);
  std::vector<clio::cte::core::Client::PrivBlobSegment> put_segs,
      get_segs;
  for (int i = 0; i < kN; ++i) {
    src[i].assign(kSeg, static_cast<char>((i % 250) + 1));
    dst[i].assign(kSeg, 0);
    put_segs.push_back(clio::cte::core::Client::PrivBlobSegment(
        static_cast<clio::run::u64>(i) * kStride, kSeg, src[i].data()));
    get_segs.push_back(clio::cte::core::Client::PrivBlobSegment(
        static_cast<clio::run::u64>(i) * kStride, kSeg, dst[i].data()));
  }

  auto pv = cte->AsyncPutBlobVectored(tag_id, blob, put_segs);
  REQUIRE(pv.get() != nullptr);
  pv.Wait();
  REQUIRE(pv->GetReturnCode() == 0);

  for (int pass = 0; pass < 2; ++pass) {  // pass 2 favors the SHM fast path
    for (auto &d : dst) std::fill(d.begin(), d.end(), 0);
    auto gv = cte->AsyncGetBlobVectored(tag_id, blob, get_segs);
    REQUIRE(gv.get() != nullptr);
    gv.Wait();
    REQUIRE(gv->GetReturnCode() == 0);
    for (int i = 0; i < kN; ++i) {
      REQUIRE(std::memcmp(dst[i].data(), src[i].data(), kSeg) == 0);
    }
  }
  std::printf("[#820] private vectored roundtrip: %d strided x %zu B OK\n",
              kN, kSeg);
}

TEST_CASE("Vectored I/O rejects malformed segments", "[cte][vectored][820]") {
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);
  auto *ipc = CLIO_IPC;
  auto *cte = CLIO_CTE_CLIENT;
  clio::cte::core::Tag tag("vec_bad_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();
  const size_t kSeg = 4096;
  auto good = FillBuf(kSeg, 'g');

  // PutBlob: a zero-size segment.
  {
    std::vector<clio::cte::core::BlobSegment> segs;
    segs.push_back(clio::cte::core::BlobSegment(0, kSeg,
                                                ctp::ipc::ShmPtr<>(good.shm_)));
    segs.push_back(clio::cte::core::BlobSegment(kSeg, 0,
                                                ctp::ipc::ShmPtr<>(good.shm_)));
    auto p = cte->AsyncPutBlobVectored(tag_id, "put_zero", segs);
    p.Wait();
    REQUIRE(p->GetReturnCode() != 0);
  }
  // PutBlob: a null-buffer segment.
  {
    std::vector<clio::cte::core::BlobSegment> segs;
    segs.push_back(clio::cte::core::BlobSegment(0, kSeg,
                                                ctp::ipc::ShmPtr<>(good.shm_)));
    segs.push_back(clio::cte::core::BlobSegment(kSeg, kSeg,
                                                ctp::ipc::ShmPtr<>::GetNull()));
    auto p = cte->AsyncPutBlobVectored(tag_id, "put_null", segs);
    p.Wait();
    REQUIRE(p->GetReturnCode() != 0);
  }
  // GetBlob: zero-size and null-buffer segments (need a blob to read).
  {
    auto p = cte->AsyncPutBlob(tag_id, "get_bad", 0, 2 * kSeg,
                               ctp::ipc::ShmPtr<>(good.shm_));
    // good only holds kSeg; that's fine, we only care the blob exists.
    p.Wait();
    auto dst = FillBuf(kSeg, 0);
    {
      std::vector<clio::cte::core::BlobSegment> segs;
      segs.push_back(clio::cte::core::BlobSegment(0, 0,
                                                  ctp::ipc::ShmPtr<>(dst.shm_)));
      auto g = cte->AsyncGetBlobVectored(tag_id, "get_bad", segs);
      g.Wait();
      REQUIRE(g->GetReturnCode() != 0);
    }
    {
      std::vector<clio::cte::core::BlobSegment> segs;
      segs.push_back(clio::cte::core::BlobSegment(0, kSeg,
                                                  ctp::ipc::ShmPtr<>::GetNull()));
      auto g = cte->AsyncGetBlobVectored(tag_id, "get_bad", segs);
      g.Wait();
      REQUIRE(g->GetReturnCode() != 0);
    }
    ipc->FreeBuffer(dst);
  }
  ipc->FreeBuffer(good);
  std::printf("[#820] vectored I/O rejects zero-size and null segments\n");
}

TEST_CASE("Vectored put/get honor the blob-transform bit",
          "[cte][vectored][818]") {
  // issue #818. The vectored APIs share PutBlobTask/GetBlobTask with the
  // scalar path, so the transform declaration must flow through them
  // identically: a vectored put that declares Context::transform_flags_ marks
  // the blob, the SHM cache record refuses direct reads, and a vectored get
  // still returns the stored bytes through the RPC fallback.
  REQUIRE(g_fixture != nullptr);
  REQUIRE(g_fixture->initialized_);
  auto *ipc = CLIO_IPC;
  auto *cte = CLIO_CTE_CLIENT;

  clio::cte::core::Tag tag("vectored_transform_tag");
  clio::cte::core::TagId tag_id = tag.GetTagId();

  const size_t kSeg = 4096;
  const int kN = 4;

  auto put_vectored = [&](const std::string &blob, clio::run::u32 transform) {
    std::vector<ctp::ipc::FullPtr<char>> bufs;
    std::vector<clio::cte::core::BlobSegment> segs;
    for (int i = 0; i < kN; ++i) {
      auto b = FillBuf(kSeg, static_cast<char>('a' + i));
      bufs.push_back(b);
      segs.push_back(clio::cte::core::BlobSegment(
          static_cast<clio::run::u64>(i) * kSeg, kSeg,
          ctp::ipc::ShmPtr<>(b.shm_)));
    }
    clio::cte::core::Context ctx;
    ctx.transform_flags_ = transform;
    auto pv = cte->AsyncPutBlobVectored(tag_id, blob, segs, /*score=*/-1.0f,
                                        ctx);
    pv.Wait();
    REQUIRE(pv->GetReturnCode() == 0);
    for (auto &b : bufs) ipc->FreeBuffer(b);
  };

  put_vectored("vt_raw", clio::cte::core::kBlobTransformNone);
  put_vectored("vt_transformed", clio::cte::core::kBlobTransformed |
                                     clio::cte::core::kBlobTransformCompressed);

  // The mark reaches the cache record, and only for the declared blob.
  if (cte->HasShmCache()) {
    clio::cte::core::ShmBlobRecord raw_rec, xf_rec;
    REQUIRE(cte->TryGetBlobRecordShm(tag_id, "vt_raw", &raw_rec));
    REQUIRE(cte->TryGetBlobRecordShm(tag_id, "vt_transformed", &xf_rec));
    REQUIRE_FALSE(raw_rec.IsTransformed());
    REQUIRE(xf_rec.IsTransformed());
    REQUIRE_FALSE(xf_rec.IsDirectReadable());
    REQUIRE((xf_rec.flags_ & clio::cte::core::kShmBlobDirectReadable) == 0);
    // CONTROL: the raw blob on the same tier IS direct-readable, so the
    // refusal above is attributable to the bit and not to placement.
    REQUIRE(raw_rec.IsDirectReadable());
  } else {
    std::printf("[#818] SHM cache unavailable; record assertions skipped\n");
  }

  // A vectored get of the transformed blob must still return the stored
  // bytes -- the guard degrades to the RPC path, never to an error.
  {
    std::vector<ctp::ipc::FullPtr<char>> bufs;
    std::vector<clio::cte::core::BlobSegment> segs;
    for (int i = 0; i < kN; ++i) {
      auto b = FillBuf(kSeg, 0);
      bufs.push_back(b);
      segs.push_back(clio::cte::core::BlobSegment(
          static_cast<clio::run::u64>(i) * kSeg, kSeg,
          ctp::ipc::ShmPtr<>(b.shm_)));
    }
    auto gv = cte->AsyncGetBlobVectored(tag_id, "vt_transformed", segs);
    gv.Wait();
    REQUIRE(gv->GetReturnCode() == 0);
    // ...and the get REPORTS the transform state back through its context, so
    // the caller can detect the bytes are stored form, not the original.
    REQUIRE((gv->context_.transform_flags_ &
             clio::cte::core::kBlobTransformed) != 0);
    for (int i = 0; i < kN; ++i) {
      std::vector<char> want(kSeg, static_cast<char>('a' + i));
      REQUIRE(std::memcmp(bufs[static_cast<size_t>(i)].ptr_, want.data(),
                          kSeg) == 0);
      ipc->FreeBuffer(bufs[static_cast<size_t>(i)]);
    }
  }
  std::printf("[#818] vectored put/get honor the transform bit\n");
}

int main(int argc, char **argv) {
  g_fixture = new Fixture();
  std::string filter = (argc > 1) ? argv[1] : "";
  int rc = SimpleTest::run_all_tests(filter);
  delete g_fixture;
  g_fixture = nullptr;
  return rc;
}
