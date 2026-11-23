/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

// Zip archives built in memory with zlib, for the dataverse:// tests.

#ifndef CAE_TEST_DATAVERSE_ZIP_BUILDER_H_
#define CAE_TEST_DATAVERSE_ZIP_BUILDER_H_

#include <zlib.h>

#include <cstdint>
#include <string>
#include <vector>

namespace zip_builder {

inline void Put16(std::string *s, uint16_t v) {
  s->push_back(static_cast<char>(v & 0xFF));
  s->push_back(static_cast<char>(v >> 8));
}
inline void Put32(std::string *s, uint32_t v) {
  Put16(s, static_cast<uint16_t>(v & 0xFFFF));
  Put16(s, static_cast<uint16_t>(v >> 16));
}
inline void Put64(std::string *s, uint64_t v) {
  Put32(s, static_cast<uint32_t>(v & 0xFFFFFFFFu));
  Put32(s, static_cast<uint32_t>(v >> 32));
}

inline std::string RawDeflate(const std::string &in) {
  z_stream zs{};
  deflateInit2(&zs, 6, Z_DEFLATED, -MAX_WBITS, 8, Z_DEFAULT_STRATEGY);
  std::string out(deflateBound(&zs, in.size()), '\0');
  zs.next_in = reinterpret_cast<Bytef *>(const_cast<char *>(in.data()));
  zs.avail_in = static_cast<uInt>(in.size());
  zs.next_out = reinterpret_cast<Bytef *>(out.data());
  zs.avail_out = static_cast<uInt>(out.size());
  deflate(&zs, Z_FINISH);
  out.resize(zs.total_out);
  deflateEnd(&zs);
  return out;
}

struct Member {
  std::string name, data;
  bool deflate;
  uint16_t flags = 0;         ///< general-purpose flags (bit 0: encrypted)
  int method = -1;            ///< compression method to record (-1: 0 or 8)
  bool bad_crc = false;       ///< record a wrong CRC-32
};

/** A zip of `ms`; with zip64, the EOCD is saturated and the sizes live in the
 *  ZIP64 records and extra fields, as large archives write them. */
inline std::string MakeZip(const std::vector<Member> &ms, bool zip64,
                    const std::string &comment = "") {
  std::string z, cd;
  for (const auto &m : ms) {
    const uint32_t crc = static_cast<uint32_t>(
        crc32(0L, reinterpret_cast<const Bytef *>(m.data.data()),
              static_cast<uInt>(m.data.size()))) ^ (m.bad_crc ? 1u : 0u);
    const uint16_t method =
        static_cast<uint16_t>(m.method >= 0 ? m.method : (m.deflate ? 8 : 0));
    const std::string payload = m.deflate ? RawDeflate(m.data) : m.data;
    const uint64_t lho = z.size();
    std::string local_extra = "XX";  // a local extra field the CD lacks
    Put32(&z, 0x04034b50);
    Put16(&z, 20);
    Put16(&z, m.flags);
    Put16(&z, method);
    Put32(&z, 0);
    Put32(&z, crc);
    Put32(&z, static_cast<uint32_t>(payload.size()));
    Put32(&z, static_cast<uint32_t>(m.data.size()));
    Put16(&z, static_cast<uint16_t>(m.name.size()));
    Put16(&z, static_cast<uint16_t>(local_extra.size()));
    z += m.name + local_extra + payload;

    std::string extra;
    if (zip64) {
      Put16(&extra, 0x0001);
      Put16(&extra, 24);
      Put64(&extra, m.data.size());
      Put64(&extra, payload.size());
      Put64(&extra, lho);
    }
    Put32(&cd, 0x02014b50);
    Put16(&cd, 45);
    Put16(&cd, 45);
    Put16(&cd, m.flags);
    Put16(&cd, method);
    Put32(&cd, 0);
    Put32(&cd, crc);
    Put32(&cd, zip64 ? 0xFFFFFFFFu : static_cast<uint32_t>(payload.size()));
    Put32(&cd, zip64 ? 0xFFFFFFFFu : static_cast<uint32_t>(m.data.size()));
    Put16(&cd, static_cast<uint16_t>(m.name.size()));
    Put16(&cd, static_cast<uint16_t>(extra.size()));
    Put16(&cd, 0);
    Put16(&cd, 0);
    Put16(&cd, 0);
    Put32(&cd, 0);
    Put32(&cd, zip64 ? 0xFFFFFFFFu : static_cast<uint32_t>(lho));
    cd += m.name + extra;
  }
  const uint64_t cd_off = z.size();
  z += cd;
  if (zip64) {
    const uint64_t rec = z.size();
    Put32(&z, 0x06064b50);
    Put64(&z, 44);
    Put16(&z, 45);
    Put16(&z, 45);
    Put32(&z, 0);
    Put32(&z, 0);
    Put64(&z, ms.size());
    Put64(&z, ms.size());
    Put64(&z, cd.size());
    Put64(&z, cd_off);
    Put32(&z, 0x07064b50);
    Put32(&z, 0);
    Put64(&z, rec);
    Put32(&z, 1);
  }
  Put32(&z, 0x06054b50);
  Put16(&z, 0);
  Put16(&z, 0);
  Put16(&z, zip64 ? 0xFFFF : static_cast<uint16_t>(ms.size()));
  Put16(&z, zip64 ? 0xFFFF : static_cast<uint16_t>(ms.size()));
  Put32(&z, zip64 ? 0xFFFFFFFFu : static_cast<uint32_t>(cd.size()));
  Put32(&z, zip64 ? 0xFFFFFFFFu : static_cast<uint32_t>(cd_off));
  Put16(&z, static_cast<uint16_t>(comment.size()));
  z += comment;
  return z;
}

}  // namespace zip_builder

#endif  // CAE_TEST_DATAVERSE_ZIP_BUILDER_H_
