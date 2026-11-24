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

#ifndef CLIO_BDEV_TLS_CA_H_
#define CLIO_BDEV_TLS_CA_H_

// Where the HTTPS clients of the cloud bdev tiers (s3_rest.h, gcs_rest.h)
// find trusted root certificates.
//
// OpenSSL's built-in default is a path fixed when OpenSSL itself was built.
// That is right for a distribution's own OpenSSL, but not for one linked into
// a portable binary: the pip wheel links a static OpenSSL built on AlmaLinux,
// whose default (/etc/pki/tls) does not exist on Debian or Ubuntu, and the
// macOS wheel bundles Homebrew's OpenSSL, whose default is under the build
// machine's Homebrew prefix. So the clients name a bundle explicitly: the
// SSL_CERT_FILE environment variable when set (the standard OpenSSL override),
// otherwise the first bundle found at the usual distribution locations.

#include <string>
#include <sys/stat.h>

#include "clio_ctp/introspect/system_info.h"

namespace clio::run::bdev {

/**
 * The CA bundle the cloud tiers' TLS context should load.
 *
 * @return SSL_CERT_FILE when it is set and non-empty; else the first existing
 *         file among the common system bundle locations (Debian/Ubuntu,
 *         RHEL/Fedora, SUSE, Alpine/macOS); else an empty string, in which
 *         case the caller relies on OpenSSL's defaults (and, on Windows,
 *         Poco's import of the system certificate store).
 */
inline std::string DefaultCaBundle() {
  std::string env = ctp::SystemInfo::Getenv("SSL_CERT_FILE");
  if (!env.empty()) return env;
  static const char *const kCandidates[] = {
      "/etc/ssl/certs/ca-certificates.crt",                 // Debian, Ubuntu
      "/etc/pki/tls/certs/ca-bundle.crt",                   // RHEL, Fedora
      "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",  // RHEL 7+
      "/etc/ssl/ca-bundle.pem",                             // SUSE
      "/etc/ssl/cert.pem",                                  // Alpine, macOS
  };
  for (const char *path : kCandidates) {
    struct stat st;
    if (stat(path, &st) == 0 && (st.st_mode & S_IFMT) == S_IFREG) return path;
  }
  return std::string();
}

}  // namespace clio::run::bdev

#endif  // CLIO_BDEV_TLS_CA_H_
