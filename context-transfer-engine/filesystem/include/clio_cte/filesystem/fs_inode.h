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

/**
 * Inode records: the attributes of one inode.
 *
 * Every inode -- file, symlink or directory -- has exactly one record, held
 * by its home container (the one its id names) and stored as one key of the
 * xattr section of the inode's own CTE tag. Names are not part of it: those
 * are rows of the inode table (fs_inode_table.h). The one exception is a
 * directory's back-pointer to the row that names it (a directory has exactly
 * one name), which rename uses to refuse moving a directory into its own
 * subtree.
 */
#ifndef CLIO_CTE_FILESYSTEM_FS_INODE_H_
#define CLIO_CTE_FILESYSTEM_FS_INODE_H_

#include <string>

#include <clio_cte/filesystem/fs_shard.h>

namespace clio::cte::filesystem {

/** Volume flag (root record): names are compared case-insensitively. */
GLOBAL_CROSS_CONST clio::run::u32 kVolumeCaseInsensitive = 1u;

/** The persisted attributes of one inode. */
struct Inode {
  clio::cte::core::TagId tag_id_;          ///< the inode's id (its tag)
  clio::run::u32 type_ = kFsTypeFile;      ///< kFsType*
  clio::run::u32 nlink_ = 1;               ///< names it has (rows)
  clio::run::u32 mode_ = 0xFFFFFFFFu;      ///< all-ones = adapter default
  clio::run::u32 uid_ = 0xFFFFFFFFu;       ///< all-ones = adapter default
  clio::run::u32 gid_ = 0xFFFFFFFFu;       ///< all-ones = adapter default
  clio::run::u64 atime_ = 0, mtime_ = 0, ctime_ = 0;  ///< ns
  std::string symlink_;                    ///< symlink target (symlinks)
  bool orphan_ = false;     ///< last name gone while open: purge at close
  bool has_xattr_ = false;  ///< an xattr blob may exist
  clio::run::u64 parent_ = 0;    ///< directories: the directory holding it
  std::string parent_key_;       ///< directories: its canonical name there
  clio::run::u32 volume_flags_ = 0;  ///< root only: kVolume* bits
  clio::run::u32 fold_version_ = 0;  ///< root only: fold table of the volume
};

/** Record format tag ("CFI2"). Records of the earlier format ("CFI1") are
 *  refused, as are the directory blocks written with them. */
GLOBAL_CROSS_CONST clio::run::u32 kInodeRecMagic = 0x32494643u;
/** Record flag: the inode is an orphan (unlinked while open). */
GLOBAL_CROSS_CONST clio::run::u32 kInodeRecOrphan = 1u;
/** Record flag: an xattr blob may exist. */
GLOBAL_CROSS_CONST clio::run::u32 kInodeRecXattr = 2u;
/** Record flag: written at create, before any access (no container caches
 *  the inode yet and no stream exists). */
GLOBAL_CROSS_CONST clio::run::u32 kInodeRecFresh = 4u;
/** DecodeInodeRecord's writer for a record that does not name one. */
GLOBAL_CROSS_CONST clio::run::u32 kNoRecWriter = 0xFFFFFFFFu;

/**
 * Encode an inode record.
 * @param ino the attributes
 * @param size logical size to record (files; 0 otherwise)
 * @param writer the container storing the record
 * @param fresh written at create by the minter (kInodeRecFresh)
 * @return record bytes
 */
inline std::string EncodeInodeRecord(const Inode &ino, clio::run::u64 size,
                                     clio::run::u32 writer, bool fresh) {
  std::string p;
  FsEnc e(&p);
  e.U32(kInodeRecMagic);
  e.U32(ino.type_);
  e.U64(size);
  e.U32(ino.nlink_);
  e.U32(ino.mode_);
  e.U32(ino.uid_);
  e.U32(ino.gid_);
  e.U64(ino.atime_);
  e.U64(ino.mtime_);
  e.U64(ino.ctime_);
  e.U32((ino.orphan_ ? kInodeRecOrphan : 0u) |
        (ino.has_xattr_ ? kInodeRecXattr : 0u) |
        (fresh ? kInodeRecFresh : 0u));
  e.Str(ino.symlink_);
  e.U32(writer);
  e.U64(ino.parent_);
  e.Str(ino.parent_key_);
  e.U32(ino.volume_flags_);
  e.U32(ino.fold_version_);
  return p;
}

/**
 * Decode an inode record (tag_id_ is left to the caller).
 * @param rec record bytes
 * @param ino receives the attributes
 * @param size receives the recorded size
 * @param writer receives the container that stored it
 * @param fresh receives the kInodeRecFresh flag (may be null)
 * @return false if malformed or of another format
 */
inline bool DecodeInodeRecord(const std::string &rec, Inode *ino,
                              clio::run::u64 *size, clio::run::u32 *writer,
                              bool *fresh) {
  FsDec d(rec.data(), rec.size());
  clio::run::u32 magic = 0, flags = 0;
  if (!d.U32(&magic) || magic != kInodeRecMagic || !d.U32(&ino->type_) ||
      !d.U64(size) || !d.U32(&ino->nlink_) || !d.U32(&ino->mode_) ||
      !d.U32(&ino->uid_) || !d.U32(&ino->gid_) || !d.U64(&ino->atime_) ||
      !d.U64(&ino->mtime_) || !d.U64(&ino->ctime_) || !d.U32(&flags) ||
      !d.Str(&ino->symlink_) || !d.U32(writer) || !d.U64(&ino->parent_) ||
      !d.Str(&ino->parent_key_) || !d.U32(&ino->volume_flags_) ||
      !d.U32(&ino->fold_version_)) {
    return false;
  }
  ino->orphan_ = (flags & kInodeRecOrphan) != 0;
  ino->has_xattr_ = (flags & kInodeRecXattr) != 0;
  if (fresh != nullptr) *fresh = (flags & kInodeRecFresh) != 0;
  return true;
}

}  // namespace clio::cte::filesystem

#endif  // CLIO_CTE_FILESYSTEM_FS_INODE_H_
