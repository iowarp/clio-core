/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

/**
 * Known-answer tests for the AWS Signature Version 4 signer (sigv4.h).
 *
 * Every expected value below was produced by botocore's SigV4Auth -- the
 * reference implementation AWS ships -- and then frozen here as a constant.
 * botocore ran once, at authoring time, via gen_sigv4_vectors.py in this
 * directory; it is not a dependency of this test. The test needs no network,
 * no credentials, no CLIO runtime and no S3 endpoint, which is the whole
 * point: signature correctness is settled here rather than by interpreting a
 * 403 from real AWS.
 *
 * To add or change a case, edit gen_sigv4_vectors.py, run it, and paste its
 * output over the kVectors table below.
 *
 * The credentials are AWS's own documentation placeholders. They are not
 * secrets and authenticate against nothing.
 */

#include "clio_runtime/bdev/transports/sigv4.h"

#include <cstring>
#include <string>

#include "simple_test.h"

namespace s3 = clio::run::bdev::s3;

namespace {

/** Fixed inputs shared by every vector. */
constexpr const char *kAccessKey = "AKIDEXAMPLE";
constexpr const char *kSecretKey = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";
constexpr const char *kSessionToken =
    "FQoDYXdzEXAMPLESESSIONTOKEN////////////";
constexpr const char *kAmzDate = "20260824T120000Z";
constexpr const char *kDateStamp = "20260824";

/** One frozen request/signature pair. */
struct Vector {
  const char *name;
  const char *method;
  const char *host;
  const char *canonical_uri;
  const char *canonical_query;
  const char *payload_hash;
  const char *region;
  const char *session_token;  ///< empty unless the case exercises STS creds
  const char *expect_signed_headers;
  const char *expect_canonical_request;
  const char *expect_string_to_sign;
  const char *expect_authorization;
};

const Vector kVectors[] = {
    {
        "get_empty_body",
        "GET", "clio-bench.s3.us-east-1.amazonaws.com",
        "/block_0",
        "", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "us-east-1", "",
        "host;x-amz-content-sha256;x-amz-date",
        "GET\n/block_0\n\nhost:clio-bench.s3.us-east-1.amazonaws.com\nx-amz-content-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\nx-amz-date:20260824T120000Z\n\nhost;x-amz-content-sha256;x-amz-date\ne3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "AWS4-HMAC-SHA256\n20260824T120000Z\n20260824/us-east-1/s3/aws4_request\n404c1c6a31275925036d6da851e6035cc1a2750875649094f0dc69ed68ed09f9",
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260824/us-east-1/s3/aws4_request, SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=23e9675941a74f34b6b10fdb494a68291967640117c7783cbf1b8a325f079f23",
    },
    {
        "put_with_body",
        "PUT", "clio-bench.s3.us-east-1.amazonaws.com",
        "/block_1048576",
        "", "05c6e08f1d9fdafa03147fcb8f82f124c76d2f70e3d989dc8aadb5e7d7450bec",
        "us-east-1", "",
        "host;x-amz-content-sha256;x-amz-date",
        "PUT\n/block_1048576\n\nhost:clio-bench.s3.us-east-1.amazonaws.com\nx-amz-content-sha256:05c6e08f1d9fdafa03147fcb8f82f124c76d2f70e3d989dc8aadb5e7d7450bec\nx-amz-date:20260824T120000Z\n\nhost;x-amz-content-sha256;x-amz-date\n05c6e08f1d9fdafa03147fcb8f82f124c76d2f70e3d989dc8aadb5e7d7450bec",
        "AWS4-HMAC-SHA256\n20260824T120000Z\n20260824/us-east-1/s3/aws4_request\nc4ad434155248aa20ad65d5d18aafb66a457cb705d1ee7fde9cb7b46931d5040",
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260824/us-east-1/s3/aws4_request, SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=e8c15fe45b801b9ff47048baa45db1e789012d154768054f309bc9b6e16be43b",
    },
    {
        "query_params",
        "GET", "clio-bench.s3.us-east-1.amazonaws.com",
        "/",
        "list-type=2&prefix=clio%2Fblocks", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "us-east-1", "",
        "host;x-amz-content-sha256;x-amz-date",
        "GET\n/\nlist-type=2&prefix=clio%2Fblocks\nhost:clio-bench.s3.us-east-1.amazonaws.com\nx-amz-content-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\nx-amz-date:20260824T120000Z\n\nhost;x-amz-content-sha256;x-amz-date\ne3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "AWS4-HMAC-SHA256\n20260824T120000Z\n20260824/us-east-1/s3/aws4_request\nb8967667d77a1d3150f1bc237173f32fee5d0ff1ec6835742df22ab3a219be10",
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260824/us-east-1/s3/aws4_request, SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=6e6ce853ae1e09e484a2f6a9c539a19c03cac27ef2472e8179bab60c022d8258",
    },
    {
        "uri_encoded_key",
        "PUT", "clio-bench.s3.us-east-1.amazonaws.com",
        "/my%20prefix/block%2B0%3D1",
        "", "05c6e08f1d9fdafa03147fcb8f82f124c76d2f70e3d989dc8aadb5e7d7450bec",
        "us-east-1", "",
        "host;x-amz-content-sha256;x-amz-date",
        "PUT\n/my%20prefix/block%2B0%3D1\n\nhost:clio-bench.s3.us-east-1.amazonaws.com\nx-amz-content-sha256:05c6e08f1d9fdafa03147fcb8f82f124c76d2f70e3d989dc8aadb5e7d7450bec\nx-amz-date:20260824T120000Z\n\nhost;x-amz-content-sha256;x-amz-date\n05c6e08f1d9fdafa03147fcb8f82f124c76d2f70e3d989dc8aadb5e7d7450bec",
        "AWS4-HMAC-SHA256\n20260824T120000Z\n20260824/us-east-1/s3/aws4_request\nc665f6d6dddff12b27eae677cc62c1ca7abea2b0a75e9404c3a9d3c5f72eb205",
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260824/us-east-1/s3/aws4_request, SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=f0ac6f67ec9b52650ec4eebbf1e19aacab17086333a743fed3c91465e436d5aa",
    },
    {
        "slashes_preserved",
        "GET", "clio-bench.s3.us-east-1.amazonaws.com",
        "/a/b/c/block_2097152",
        "", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "us-east-1", "",
        "host;x-amz-content-sha256;x-amz-date",
        "GET\n/a/b/c/block_2097152\n\nhost:clio-bench.s3.us-east-1.amazonaws.com\nx-amz-content-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\nx-amz-date:20260824T120000Z\n\nhost;x-amz-content-sha256;x-amz-date\ne3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "AWS4-HMAC-SHA256\n20260824T120000Z\n20260824/us-east-1/s3/aws4_request\n93f75696df5da6fdf4fcbfdbb617484a7597167bd2581137550702e70e240e64",
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260824/us-east-1/s3/aws4_request, SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=0cfe3028d0961b1f01713f47225486f3366d28a1a66e54461b4aa34dbc60b87f",
    },
    {
        "unsigned_payload",
        "PUT", "clio-bench.s3.us-east-1.amazonaws.com",
        "/block_3145728",
        "", "UNSIGNED-PAYLOAD",
        "us-east-1", "",
        "host;x-amz-content-sha256;x-amz-date",
        "PUT\n/block_3145728\n\nhost:clio-bench.s3.us-east-1.amazonaws.com\nx-amz-content-sha256:UNSIGNED-PAYLOAD\nx-amz-date:20260824T120000Z\n\nhost;x-amz-content-sha256;x-amz-date\nUNSIGNED-PAYLOAD",
        "AWS4-HMAC-SHA256\n20260824T120000Z\n20260824/us-east-1/s3/aws4_request\n043d8093e06b7cbbb62693f3f98cf67fedbb868eb2f6aa5f4a6e436a90fd642e",
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260824/us-east-1/s3/aws4_request, SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=d1dbd0f3825fbcdd9ae6bf0ee48c24ce991d00650503f555b616d0e183cc2162",
    },
    {
        "session_token",
        "GET", "clio-bench.s3.us-east-1.amazonaws.com",
        "/block_4194304",
        "", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "us-east-1", "FQoDYXdzEXAMPLESESSIONTOKEN////////////",
        "host;x-amz-content-sha256;x-amz-date;x-amz-security-token",
        "GET\n/block_4194304\n\nhost:clio-bench.s3.us-east-1.amazonaws.com\nx-amz-content-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\nx-amz-date:20260824T120000Z\nx-amz-security-token:FQoDYXdzEXAMPLESESSIONTOKEN////////////\n\nhost;x-amz-content-sha256;x-amz-date;x-amz-security-token\ne3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "AWS4-HMAC-SHA256\n20260824T120000Z\n20260824/us-east-1/s3/aws4_request\n45937914b382cae90e2438909357ad01fc9e027898b03cd939d3d71bb846efd3",
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260824/us-east-1/s3/aws4_request, SignedHeaders=host;x-amz-content-sha256;x-amz-date;x-amz-security-token, Signature=55750a1f42e7103182dcab78005e365d126bc217bde353c6dbdc4bf5d9acc87d",
    },
    {
        "region_us_east_2",
        "PUT", "clio-bench.s3.us-east-2.amazonaws.com",
        "/clio/block_0",
        "", "UNSIGNED-PAYLOAD",
        "us-east-2", "",
        "host;x-amz-content-sha256;x-amz-date",
        "PUT\n/clio/block_0\n\nhost:clio-bench.s3.us-east-2.amazonaws.com\nx-amz-content-sha256:UNSIGNED-PAYLOAD\nx-amz-date:20260824T120000Z\n\nhost;x-amz-content-sha256;x-amz-date\nUNSIGNED-PAYLOAD",
        "AWS4-HMAC-SHA256\n20260824T120000Z\n20260824/us-east-2/s3/aws4_request\n4e27bda24963a7d631f992ad459b02db1a42d452037536c7425ec217428a7dea",
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260824/us-east-2/s3/aws4_request, SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=e5ea86525cc6bdb2103d7d4371683a4591409e3b93815b58f0da0eb9fe43abce",
    },
    {
        "path_style_endpoint",
        "DELETE", "localhost:9000",
        "/clio-bench/clio/block_0",
        "", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "us-east-2", "",
        "host;x-amz-content-sha256;x-amz-date",
        "DELETE\n/clio-bench/clio/block_0\n\nhost:localhost:9000\nx-amz-content-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\nx-amz-date:20260824T120000Z\n\nhost;x-amz-content-sha256;x-amz-date\ne3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "AWS4-HMAC-SHA256\n20260824T120000Z\n20260824/us-east-2/s3/aws4_request\n6c37d88b6ff13ed760844e3d20eeefffd9c6dbccefc4f37e8327cc5a614f3226",
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260824/us-east-2/s3/aws4_request, SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=90a5dadff23e29b1c5a7cd99c86a5ebdfca48df650901f8162c8722fb49d3377",
    },
    {
        "head_bucket",
        "HEAD", "clio-bench.s3.us-east-2.amazonaws.com",
        "/",
        "", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "us-east-2", "",
        "host;x-amz-content-sha256;x-amz-date",
        "HEAD\n/\n\nhost:clio-bench.s3.us-east-2.amazonaws.com\nx-amz-content-sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\nx-amz-date:20260824T120000Z\n\nhost;x-amz-content-sha256;x-amz-date\ne3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "AWS4-HMAC-SHA256\n20260824T120000Z\n20260824/us-east-2/s3/aws4_request\n72f80750104e0c4f2ac2e4d52c11f096e6664e4d9a0c27ac1df34f93cb55949f",
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260824/us-east-2/s3/aws4_request, SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=721eab0c47d562a85a5697f62be535a36e1643ea93bf70ea15ab59e97d1175f1",
    },
};

}  // namespace

TEST_CASE("sigv4_primitives", "[sigv4]") {
  SECTION("SHA-256");
  REQUIRE(s3::Sha256Hex(std::string("")) == std::string(s3::kEmptySha256));
  REQUIRE(s3::Sha256Hex(std::string("abc")) ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  REQUIRE(s3::Sha256Hex("the quick brown fox jumps over the lazy dog", 43) ==
          "05c6e08f1d9fdafa03147fcb8f82f124c76d2f70e3d989dc8aadb5e7d7450bec");

  SECTION("HMAC-SHA256 (RFC 4231 style)");
  REQUIRE(s3::HmacSha256Hex("key",
                            "The quick brown fox jumps over the lazy dog") ==
          "f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8");

  SECTION("signing-key derivation");
  std::string key = s3::DeriveSigningKey(kSecretKey, kDateStamp, "us-east-2");
  REQUIRE(key.size() == 32);
  REQUIRE(Poco::DigestEngine::digestToHex(Poco::DigestEngine::Digest(
              key.begin(), key.end())) ==
          "53c67bae86da713624a54a1d300704dfea67be625e3e65782cc93678556aa580");
}

TEST_CASE("sigv4_uri_encode", "[sigv4]") {
  SECTION("unreserved characters pass through");
  REQUIRE(s3::UriEncode("abcXYZ019-_.~", true) == "abcXYZ019-_.~");

  SECTION("slash is kept or encoded per keep_slash");
  REQUIRE(s3::UriEncode("a/b/c", true) == "a/b/c");
  REQUIRE(s3::UriEncode("a/b/c", false) == "a%2Fb%2Fc");

  SECTION("reserved characters are percent-encoded, uppercase hex");
  REQUIRE(s3::UriEncode("my prefix", true) == "my%20prefix");
  REQUIRE(s3::UriEncode("block+0=1", true) == "block%2B0%3D1");

  SECTION("high bytes encode per byte, not per character");
  REQUIRE(s3::UriEncode("\xc3\xa9", true) == "%C3%A9");
}

TEST_CASE("sigv4_known_answer_vectors", "[sigv4]") {
  const size_t count = sizeof(kVectors) / sizeof(kVectors[0]);
  REQUIRE(count >= 8);
  for (size_t i = 0; i < count; ++i) {
    const Vector &v = kVectors[i];
    SECTION(v.name);

    s3::SigV4Credentials creds;
    creds.access_key = kAccessKey;
    creds.secret_key = kSecretKey;
    creds.region = v.region;
    creds.session_token = v.session_token;

    s3::SigningInput in;
    in.method = v.method;
    in.canonical_uri = v.canonical_uri;
    in.canonical_query = v.canonical_query;
    in.host = v.host;
    in.payload_hash = v.payload_hash;

    s3::AmzTime t;
    t.amz_date = kAmzDate;
    t.date_stamp = kDateStamp;

    // Compared stage by stage: when a signature is wrong, the first mismatched
    // stage says which part of the canonical form is at fault, which is exactly
    // the diagnostic a 403 from S3 does not give you.
    REQUIRE(s3::SignedHeaders(creds) == std::string(v.expect_signed_headers));
    std::string cr = s3::CanonicalRequest(in, creds, t);
    REQUIRE(cr == std::string(v.expect_canonical_request));
    REQUIRE(s3::StringToSign(cr, creds, t) ==
            std::string(v.expect_string_to_sign));
    REQUIRE(s3::BuildAuthorization(in, creds, t) ==
            std::string(v.expect_authorization));
  }
}

TEST_CASE("sigv4_signature_is_sensitive_to_every_input", "[sigv4]") {
  // A signer that ignored one of its inputs would still pass the vectors above
  // if that input happened to be constant across them. Perturb each field of a
  // known-good request and require the signature to move.
  const Vector &base = kVectors[0];
  s3::SigV4Credentials creds;
  creds.access_key = kAccessKey;
  creds.secret_key = kSecretKey;
  creds.region = base.region;

  s3::SigningInput in;
  in.method = base.method;
  in.canonical_uri = base.canonical_uri;
  in.canonical_query = base.canonical_query;
  in.host = base.host;
  in.payload_hash = base.payload_hash;

  s3::AmzTime t;
  t.amz_date = kAmzDate;
  t.date_stamp = kDateStamp;

  const std::string good = s3::BuildAuthorization(in, creds, t);
  REQUIRE(good == std::string(base.expect_authorization));

  SECTION("method");
  s3::SigningInput m = in;
  m.method = "HEAD";
  REQUIRE(s3::BuildAuthorization(m, creds, t) != good);

  SECTION("path");
  s3::SigningInput p = in;
  p.canonical_uri = "/block_1";
  REQUIRE(s3::BuildAuthorization(p, creds, t) != good);

  SECTION("host");
  s3::SigningInput h = in;
  h.host = "other.s3.us-east-1.amazonaws.com";
  REQUIRE(s3::BuildAuthorization(h, creds, t) != good);

  SECTION("payload hash");
  s3::SigningInput q = in;
  q.payload_hash = s3::kUnsignedPayload;
  REQUIRE(s3::BuildAuthorization(q, creds, t) != good);

  SECTION("region");
  s3::SigV4Credentials r = creds;
  r.region = "us-east-2";
  REQUIRE(s3::BuildAuthorization(in, r, t) != good);

  SECTION("secret key");
  s3::SigV4Credentials k = creds;
  k.secret_key = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEZ";
  REQUIRE(s3::BuildAuthorization(in, k, t) != good);

  SECTION("session token changes SignedHeaders and the signature");
  s3::SigV4Credentials s = creds;
  s.session_token = kSessionToken;
  REQUIRE(s3::SignedHeaders(s) ==
          "host;x-amz-content-sha256;x-amz-date;x-amz-security-token");
  REQUIRE(s3::BuildAuthorization(in, s, t) != good);

  SECTION("timestamp");
  s3::AmzTime t2;
  t2.amz_date = "20260825T120000Z";
  t2.date_stamp = "20260825";
  REQUIRE(s3::BuildAuthorization(in, creds, t2) != good);
}

TEST_CASE("sigv4_now_amz_time_is_well_formed", "[sigv4]") {
  s3::AmzTime t = s3::NowAmzTime();
  REQUIRE(t.amz_date.size() == 16);
  REQUIRE(t.amz_date[8] == 'T');
  REQUIRE(t.amz_date[15] == 'Z');
  REQUIRE(t.date_stamp.size() == 8);
  REQUIRE(t.amz_date.compare(0, 8, t.date_stamp) == 0);
}

SIMPLE_TEST_MAIN()
