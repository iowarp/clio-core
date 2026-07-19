/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SAFE_BDEV_SUPERBLOCK_H_
#define SAFE_BDEV_SUPERBLOCK_H_

#include <cstddef>
#include <cstdint>

/**
 * Member-superblock layout for safe_bdev.
 *
 * The first 64 KiB (kSuperblockSize) of every member bdev — at absolute offset
 * 0 — is reserved for a fixed-layout member superblock that records the array
 * identity (the safe_bdev pool id) and this member's role within it. DATA shards
 * begin at offset kSuperblockSize. The superblock lets Create distinguish a
 * fresh (blank) member from a member that already belongs to THIS array (a
 * restart / re-attach) from a FOREIGN member that another safe_bdev already
 * owns (which must be refused so its data is not clobbered).
 *
 * MemberSuperblock is a plain trivially-copyable POD written as raw bytes into
 * a zero-padded kSuperblockSize buffer; sizeof(MemberSuperblock) must stay <=
 * kSuperblockSize. A blank RAM member reads back as all zeros => magic 0 =>
 * not present.
 */

namespace clio::run::safe_bdev {

/** "SAFEBDV" — fixed magic identifying a member superblock. */
static constexpr uint64_t kMemberSuperblockMagic = 0x53414645424456ULL;

/** Current on-disk format version of the member superblock. */
static constexpr uint32_t kMemberSuperblockVersion = 1;

/**
 * Fixed-layout member superblock POD. Written as raw little-endian bytes; not
 * cereal-serialized. All multi-byte fields are written in host byte order
 * (members are local-process RAM/file bdevs in the current cut). The trailing
 * checksum is a simple additive sum over every preceding byte.
 */
struct MemberSuperblock {
  uint64_t magic;           // == kMemberSuperblockMagic when present
  uint32_t format_version;  // == kMemberSuperblockVersion
  uint32_t flags;           // reserved (0)
  uint64_t array_major;     // safe_bdev pool id major (array identity)
  uint64_t array_minor;     // safe_bdev pool id minor (array identity)
  uint32_t member_slot;     // global member id within the array
  uint32_t role;            // ec::EcRole (0=data, 1=parity)
  uint32_t index;           // role index (data column / parity row hint)
  uint32_t max_failures;    // array fault-tolerance target (M)
  uint64_t shard_len;       // per-member shard length in bytes
  uint64_t epoch;           // generation/epoch at write time
  uint32_t checksum;        // additive sum over all preceding bytes

  MemberSuperblock()
      : magic(0),
        format_version(0),
        flags(0),
        array_major(0),
        array_minor(0),
        member_slot(0),
        role(0),
        index(0),
        max_failures(0),
        shard_len(0),
        epoch(0),
        checksum(0) {}

  /** Additive checksum over every byte preceding the checksum field. */
  uint32_t ComputeChecksum() const {
    const auto *bytes = reinterpret_cast<const uint8_t *>(this);
    const size_t n = offsetof(MemberSuperblock, checksum);
    uint32_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
      sum += bytes[i];
    }
    return sum;
  }

  /** True if this superblock is well-formed (magic + checksum match). */
  bool Validate() const {
    return magic == kMemberSuperblockMagic &&
           format_version == kMemberSuperblockVersion &&
           checksum == ComputeChecksum();
  }
};

static_assert(sizeof(MemberSuperblock) <= 65536,
              "MemberSuperblock must fit within the reserved superblock area");

}  // namespace clio::run::safe_bdev

#endif  // SAFE_BDEV_SUPERBLOCK_H_
