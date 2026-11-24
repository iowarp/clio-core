/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

#ifndef CLIO_BDEV_SIGV4_H_
#define CLIO_BDEV_SIGV4_H_

// AWS Signature Version 4 request signing, as pure functions.
//
// Deliberately depends on Poco *Foundation* only -- no Poco::Net, no sockets,
// no environment, no globals. Everything here is strings in, strings out, so
// correctness is provable by known-answer vectors in an ordinary unit test
// rather than by a successful request against real AWS. test_sigv4.cc holds
// those vectors, generated once from botocore's SigV4Auth (the reference
// implementation) and frozen as constants.
//
// This header carries no CLIO_ENABLE_AMAZON_DRIVE guard on purpose: it is
// feature-flag independent, which is what lets the test build and run without
// the S3 bdev, Poco::Net, or any credentials being present.

// BLOCK_SIZE / DIGEST_SIZE are the member names Poco::HMACEngine reads off its
// template argument, so neither this header nor Poco's can rename them -- but
// <linux/fs.h> (reached via <liburing.h>, which fs_bdev_transport.h includes)
// defines BLOCK_SIZE as `(1<<BLOCK_SIZE_BITS)`. Any translation unit that pulls
// in the io_uring path before this one therefore turns both enumerators into
// `(1<<10) = ...` and the parse collapses. Drop the macros for the span that
// needs the identifiers and hand them back afterwards, so a TU that includes
// this header first still sees whatever <linux/fs.h> defines later.
#pragma push_macro("BLOCK_SIZE")
#pragma push_macro("DIGEST_SIZE")
#undef BLOCK_SIZE
#undef DIGEST_SIZE

#include <Poco/DigestEngine.h>
#include <Poco/HMACEngine.h>
#include <Poco/SHA2Engine.h>

#include <cstddef>
#include <ctime>
#include <string>

namespace clio::run::bdev::s3 {

/** x-amz-content-sha256 value that opts out of payload signing. */
inline constexpr const char *kUnsignedPayload = "UNSIGNED-PAYLOAD";

/** SHA-256 of the empty string; the payload hash for bodyless requests. */
inline constexpr const char *kEmptySha256 =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/**
 * Percent-encode per RFC 3986, keeping the unreserved set. With `keep_slash`
 * true '/' passes through, which is what a canonical URI path needs; otherwise
 * '/' becomes %2F (used when encoding a single path segment).
 */
inline std::string UriEncode(const std::string &value, bool keep_slash) {
  static const char *hex = "0123456789ABCDEF";
  std::string out;
  out.reserve(value.size() * 3);
  for (unsigned char c : value) {
    bool unreserved = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                      (c >= '0' && c <= '9') || c == '-' || c == '_' ||
                      c == '.' || c == '~';
    if (unreserved || (keep_slash && c == '/')) {
      out.push_back(static_cast<char>(c));
    } else {
      out.push_back('%');
      out.push_back(hex[(c >> 4) & 0xF]);
      out.push_back(hex[c & 0xF]);
    }
  }
  return out;
}

/**
 * Poco::SHA2Engine pinned to SHA-256 and given the BLOCK_SIZE / DIGEST_SIZE
 * enums Poco::HMACEngine reads off its template argument.
 *
 * Two Poco quirks make this adapter necessary: Poco::Crypto::DigestEngine
 * cannot be HMACEngine's argument at all (it needs an algorithm name in its
 * constructor, while HMACEngine default-constructs its engine), and
 * Poco::SHA2Engine -- which *is* default-constructible -- does not publish the
 * two size enums that SHA1Engine and MD5Engine do.
 */
class Sha256Engine : public Poco::SHA2Engine {
 public:
  enum {
    BLOCK_SIZE = 64,  ///< SHA-256 input block size, bytes (RFC 6234)
    DIGEST_SIZE = 32  ///< SHA-256 output size, bytes
  };
  Sha256Engine() : Poco::SHA2Engine(Poco::SHA2Engine::SHA_256) {}
};

/** HMAC-SHA256 engine usable through Poco's DigestEngine interface. */
using HmacSha256Engine = Poco::HMACEngine<Sha256Engine>;

/** Raw bytes of a digest as a std::string (may contain embedded NULs). */
inline std::string DigestToBytes(const Poco::DigestEngine::Digest &d) {
  return std::string(reinterpret_cast<const char *>(d.data()), d.size());
}

/** Lowercase hex SHA-256 of a buffer. */
inline std::string Sha256Hex(const char *data, size_t len) {
  Sha256Engine sha;
  if (len > 0 && data != nullptr) {
    sha.update(data, static_cast<unsigned>(len));
  }
  return Poco::DigestEngine::digestToHex(sha.digest());
}

/** Lowercase hex SHA-256 of a string. */
inline std::string Sha256Hex(const std::string &s) {
  return Sha256Hex(s.data(), s.size());
}

/** HMAC-SHA256(key, msg) as raw bytes, so calls can be chained. */
inline std::string HmacSha256(const std::string &key, const std::string &msg) {
  HmacSha256Engine hmac(key.data(), key.size());
  hmac.update(msg);
  return DigestToBytes(hmac.digest());
}

/** HMAC-SHA256(key, msg) as lowercase hex. */
inline std::string HmacSha256Hex(const std::string &key,
                                 const std::string &msg) {
  HmacSha256Engine hmac(key.data(), key.size());
  hmac.update(msg);
  return Poco::DigestEngine::digestToHex(hmac.digest());
}

/**
 * Derive the SigV4 signing key:
 *   HMAC(HMAC(HMAC(HMAC("AWS4"+secret, date), region), "s3"), "aws4_request")
 * @param secret AWS_SECRET_ACCESS_KEY
 * @param date   Date stamp, YYYYMMDD (UTC)
 * @param region AWS region, e.g. us-east-1
 * @return The 32-byte signing key as raw bytes.
 */
inline std::string DeriveSigningKey(const std::string &secret,
                                    const std::string &date,
                                    const std::string &region) {
  std::string k_date = HmacSha256("AWS4" + secret, date);
  std::string k_region = HmacSha256(k_date, region);
  std::string k_service = HmacSha256(k_region, "s3");
  return HmacSha256(k_service, "aws4_request");
}

/** The two UTC timestamps every signed request needs. */
struct AmzTime {
  std::string amz_date;    ///< YYYYMMDDTHHMMSSZ, the x-amz-date header value
  std::string date_stamp;  ///< YYYYMMDD, the credential-scope date
};

/** Current UTC time formatted for SigV4. */
inline AmzTime NowAmzTime() {
  AmzTime t;
  std::time_t now = std::time(nullptr);
  std::tm tm_utc{};
#if defined(_WIN32)
  gmtime_s(&tm_utc, &now);
#else
  gmtime_r(&now, &tm_utc);
#endif
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y%m%dT%H%M%SZ", &tm_utc);
  t.amz_date = buf;
  std::strftime(buf, sizeof(buf), "%Y%m%d", &tm_utc);
  t.date_stamp = buf;
  return t;
}

/** Credentials and region scope for signing. */
struct SigV4Credentials {
  std::string access_key;     ///< AWS_ACCESS_KEY_ID
  std::string secret_key;     ///< AWS_SECRET_ACCESS_KEY
  std::string region;         ///< AWS region, e.g. us-east-2
  std::string session_token;  ///< AWS_SESSION_TOKEN; empty for long-term creds
};

/** One request, described in exactly the terms the signer needs. */
struct SigningInput {
  std::string method;           ///< GET / PUT / DELETE / HEAD
  std::string canonical_uri;    ///< Already URI-encoded path, leading '/'
  std::string canonical_query;  ///< Already encoded and sorted; usually empty
  std::string host;             ///< Exactly the Host header sent on the wire
  std::string payload_hash;     ///< x-amz-content-sha256 value
};

/**
 * The SignedHeaders list for a request, in the canonical lowercase-sorted
 * order. Only the headers S3 requires are signed: host, x-amz-content-sha256,
 * x-amz-date, plus x-amz-security-token with temporary credentials. Extra
 * unsigned headers (Content-Length, Content-Type) are permitted by S3 and
 * deliberately excluded.
 */
inline std::string SignedHeaders(const SigV4Credentials &creds) {
  std::string signed_headers = "host;x-amz-content-sha256;x-amz-date";
  if (!creds.session_token.empty()) {
    signed_headers += ";x-amz-security-token";
  }
  return signed_headers;
}

/** Build the SigV4 canonical request. The header block is already sorted. */
inline std::string CanonicalRequest(const SigningInput &in,
                                    const SigV4Credentials &creds,
                                    const AmzTime &t) {
  std::string canonical_headers = "host:" + in.host + "\n" +
                                  "x-amz-content-sha256:" + in.payload_hash +
                                  "\n" + "x-amz-date:" + t.amz_date + "\n";
  if (!creds.session_token.empty()) {
    canonical_headers += "x-amz-security-token:" + creds.session_token + "\n";
  }
  return in.method + "\n" + in.canonical_uri + "\n" + in.canonical_query +
         "\n" + canonical_headers + "\n" + SignedHeaders(creds) + "\n" +
         in.payload_hash;
}

/** The credential scope: `<date>/<region>/s3/aws4_request`. */
inline std::string CredentialScope(const SigV4Credentials &creds,
                                   const AmzTime &t) {
  return t.date_stamp + "/" + creds.region + "/s3/aws4_request";
}

/** Build the string-to-sign from an already-built canonical request. */
inline std::string StringToSign(const std::string &canonical_request,
                                const SigV4Credentials &creds,
                                const AmzTime &t) {
  return "AWS4-HMAC-SHA256\n" + t.amz_date + "\n" + CredentialScope(creds, t) +
         "\n" + Sha256Hex(canonical_request);
}

/**
 * Build the complete `Authorization:` header value for one request.
 * @param in    Request description; `canonical_uri` must already be encoded.
 * @param creds Credentials and region.
 * @param t     UTC timestamps for this request.
 * @return The header value, ready to set on the request.
 */
inline std::string BuildAuthorization(const SigningInput &in,
                                      const SigV4Credentials &creds,
                                      const AmzTime &t) {
  std::string string_to_sign =
      StringToSign(CanonicalRequest(in, creds, t), creds, t);
  std::string signing_key =
      DeriveSigningKey(creds.secret_key, t.date_stamp, creds.region);
  std::string signature = HmacSha256Hex(signing_key, string_to_sign);
  return "AWS4-HMAC-SHA256 Credential=" + creds.access_key + "/" +
         CredentialScope(creds, t) + ", SignedHeaders=" + SignedHeaders(creds) +
         ", Signature=" + signature;
}

}  // namespace clio::run::bdev::s3

// Restore whatever the including TU had; see the push near the top. Safe to do
// here because macro expansion already happened when Poco::HMACEngine's body
// and Sha256Engine's enum were tokenized -- instantiating the template later
// does not re-expand them.
#pragma pop_macro("DIGEST_SIZE")
#pragma pop_macro("BLOCK_SIZE")

#endif  // CLIO_BDEV_SIGV4_H_
