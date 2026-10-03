/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 */

/**
 * Unit tests for the in-process AWS credential/region resolver
 * (clio_cae/core/factory/aws_creds.h). Pure functions, no runtime, no network,
 * no real ~/.aws -- every input is injected via ResolveAwsCredentialsFrom, so
 * the precedence rules are pinned deterministically.
 */

#include <clio_cae/core/factory/aws_creds.h>

#include <string>

#include "simple_test.h"

namespace cae = clio::cae::core;

namespace {

const char* kCreds =
    "[default]\n"
    "aws_access_key_id = AKIA_DEFAULT\n"
    "aws_secret_access_key = SECRET_DEFAULT\n"
    "\n"
    "[bench]\n"
    "aws_access_key_id = AKIA_BENCH\n"
    "aws_secret_access_key = SECRET_BENCH\n"
    "aws_session_token = TOKEN_BENCH\n";

const char* kConfig =
    "[default]\n"
    "region = us-east-1\n"
    "\n"
    "[profile bench]\n"
    "region = us-east-2\n";

}  // namespace

TEST_CASE("aws_creds_env_keys_win_over_profile", "[aws_creds]") {
  cae::AwsCredEnv env;
  env.access_key = "AKIA_ENV";
  env.secret_key = "SECRET_ENV";
  env.session_token = "TOKEN_ENV";
  env.region = "eu-west-1";
  // ctx names a profile, but real env keys must take precedence over the file.
  cae::AwsCredResult r =
      cae::ResolveAwsCredentialsFrom("bench", "", env, kCreds, kConfig);
  REQUIRE(r.ok);
  REQUIRE(r.creds.access_key == "AKIA_ENV");
  REQUIRE(r.creds.secret_key == "SECRET_ENV");
  REQUIRE(r.creds.session_token == "TOKEN_ENV");
  REQUIRE(r.creds.region == "eu-west-1");
}

TEST_CASE("aws_creds_ctx_profile_selects_credentials", "[aws_creds]") {
  cae::AwsCredEnv env;  // no env keys -> fall to the file
  cae::AwsCredResult r =
      cae::ResolveAwsCredentialsFrom("bench", "us-west-2", env, kCreds, kConfig);
  REQUIRE(r.ok);
  REQUIRE(r.creds.access_key == "AKIA_BENCH");
  REQUIRE(r.creds.secret_key == "SECRET_BENCH");
  REQUIRE(r.creds.session_token == "TOKEN_BENCH");
  REQUIRE(r.creds.region == "us-west-2");  // ctx region wins
}

TEST_CASE("aws_creds_default_profile_when_none_named", "[aws_creds]") {
  cae::AwsCredEnv env;
  // No ctx profile, no AWS_PROFILE -> "default"; region from the config file.
  cae::AwsCredResult r =
      cae::ResolveAwsCredentialsFrom("", "", env, kCreds, kConfig);
  REQUIRE(r.ok);
  REQUIRE(r.creds.access_key == "AKIA_DEFAULT");
  REQUIRE(r.creds.region == "us-east-1");
  REQUIRE(r.creds.session_token.empty());
}

TEST_CASE("aws_creds_env_profile_used_when_ctx_empty", "[aws_creds]") {
  cae::AwsCredEnv env;
  env.profile = "bench";
  cae::AwsCredResult r =
      cae::ResolveAwsCredentialsFrom("", "", env, kCreds, kConfig);
  REQUIRE(r.ok);
  REQUIRE(r.creds.access_key == "AKIA_BENCH");
  REQUIRE(r.creds.region == "us-east-2");  // "[profile bench]" in config
}

TEST_CASE("aws_creds_region_precedence_order", "[aws_creds]") {
  cae::AwsCredEnv env;
  env.access_key = "A";
  env.secret_key = "B";
  env.region = "region-default-env";    // AWS_DEFAULT_REGION
  env.region_alt = "region-alt-env";    // AWS_REGION (lower precedence)
  // AWS_DEFAULT_REGION beats AWS_REGION.
  cae::AwsCredResult r1 =
      cae::ResolveAwsCredentialsFrom("", "", env, "", "");
  REQUIRE(r1.ok);
  REQUIRE(r1.creds.region == "region-default-env");
  // ctx region beats everything.
  cae::AwsCredResult r2 =
      cae::ResolveAwsCredentialsFrom("", "ctx-region", env, "", "");
  REQUIRE(r2.creds.region == "ctx-region");
  // Only AWS_REGION present -> it is used.
  env.region.clear();
  cae::AwsCredResult r3 =
      cae::ResolveAwsCredentialsFrom("", "", env, "", "");
  REQUIRE(r3.creds.region == "region-alt-env");
}

TEST_CASE("aws_creds_missing_profile_is_a_named_error", "[aws_creds]") {
  cae::AwsCredEnv env;
  cae::AwsCredResult r =
      cae::ResolveAwsCredentialsFrom("nope", "us-east-1", env, kCreds, kConfig);
  REQUIRE_FALSE(r.ok);
  REQUIRE(r.error.find("nope") != std::string::npos);
}

TEST_CASE("aws_creds_region_unresolved_is_an_error", "[aws_creds]") {
  cae::AwsCredEnv env;
  env.access_key = "A";
  env.secret_key = "B";
  // No ctx/env/config region anywhere -> loud error (no silent us-east-1).
  cae::AwsCredResult r =
      cae::ResolveAwsCredentialsFrom("", "", env, "", "");
  REQUIRE_FALSE(r.ok);
  REQUIRE(r.error.find("region") != std::string::npos);
}

TEST_CASE("aws_creds_profile_missing_secret_is_an_error", "[aws_creds]") {
  cae::AwsCredEnv env;
  const char* partial = "[bench]\naws_access_key_id = AKIA_ONLY\n";
  cae::AwsCredResult r =
      cae::ResolveAwsCredentialsFrom("bench", "us-east-1", env, partial, "");
  REQUIRE_FALSE(r.ok);
  REQUIRE(r.error.find("bench") != std::string::npos);
}

TEST_CASE("aws_creds_secret_never_appears_in_error_text", "[aws_creds]") {
  cae::AwsCredEnv env;
  // Force a region failure while credentials resolved from the file, and prove
  // the diagnostic text does not leak the secret it just read.
  cae::AwsCredResult r =
      cae::ResolveAwsCredentialsFrom("bench", "", env, kCreds, "");
  REQUIRE_FALSE(r.ok);  // region unresolved
  REQUIRE(r.error.find("SECRET_BENCH") == std::string::npos);
  REQUIRE(r.error.find("TOKEN_BENCH") == std::string::npos);
}

SIMPLE_TEST_MAIN()
