#!/usr/bin/env python3
"""Regenerate the frozen SigV4 known-answer vectors in test_sigv4.cc.

botocore's SigV4Auth is the reference implementation AWS ships, so it -- not
our own reading of the spec -- decides what a correct signature is. This script
drives it once, at authoring time, and prints a C++ initializer list. The
result is pasted into the `kVectors` table in test_sigv4.cc, so the committed
test has no Python, network, or credential dependency.

Usage:
    python3 -m venv /tmp/botovenv && /tmp/botovenv/bin/pip install botocore
    /tmp/botovenv/bin/python gen_sigv4_vectors.py

The credentials below are AWS's own documentation placeholders. They are not
secrets and authenticate against nothing.

Note on how botocore is driven: `SigV4Auth.add_auth` stamps the request with
`datetime.utcnow()`, which would make the output non-reproducible. The signing
stages are therefore called directly with `request.context['timestamp']` pinned.
S3SigV4Auth (not SigV4Auth) is used because S3 must NOT normalize the URL path,
and because it takes the payload hash from the X-Amz-Content-SHA256 header --
which is what lets the UNSIGNED-PAYLOAD case be expressed at all.
"""

import hashlib
import json

from botocore.auth import S3SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials

ACCESS_KEY = "AKIDEXAMPLE"
SECRET_KEY = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
SESSION_TOKEN = "FQoDYXdzEXAMPLESESSIONTOKEN////////////"
AMZ_DATE = "20260824T120000Z"

EMPTY_SHA = hashlib.sha256(b"").hexdigest()
UNSIGNED = "UNSIGNED-PAYLOAD"
BODY_SHA = hashlib.sha256(
    b"the quick brown fox jumps over the lazy dog").hexdigest()


def sign(name, method, host, uri, query, payload, region, token=""):
    """Sign one request with botocore and return every intermediate stage."""
    creds = Credentials(ACCESS_KEY, SECRET_KEY, token or None)
    url = f"https://{host}{uri}" + (f"?{query}" if query else "")
    req = AWSRequest(method=method, url=url)
    req.headers["Host"] = host
    req.headers["X-Amz-Date"] = AMZ_DATE
    req.headers["X-Amz-Content-SHA256"] = payload
    if token:
        req.headers["X-Amz-Security-Token"] = token
    req.context["timestamp"] = AMZ_DATE

    auth = S3SigV4Auth(creds, "s3", region)
    cr = auth.canonical_request(req)
    sts = auth.string_to_sign(req, cr)
    sig = auth.signature(sts, req)
    signed_headers = auth.signed_headers(auth.headers_to_sign(req))
    authorization = (
        f"AWS4-HMAC-SHA256 Credential={auth.scope(req)}, "
        f"SignedHeaders={signed_headers}, Signature={sig}"
    )
    return [name, method, host, uri, query, payload, region, token,
            signed_headers, cr, sts, authorization]


AWS1 = "clio-bench.s3.us-east-1.amazonaws.com"
AWS2 = "clio-bench.s3.us-east-2.amazonaws.com"

CASES = [
    # The block keys the S3 bdev actually writes: bodyless GET and PUT+body.
    sign("get_empty_body", "GET", AWS1, "/block_0", "", EMPTY_SHA, "us-east-1"),
    sign("put_with_body", "PUT", AWS1, "/block_1048576", "", BODY_SHA,
         "us-east-1"),
    # A non-empty canonical query string must land on its own line.
    sign("query_params", "GET", AWS1, "/", "list-type=2&prefix=clio%2Fblocks",
         EMPTY_SHA, "us-east-1"),
    # Percent-encoding of the canonical URI, and '/' surviving it.
    sign("uri_encoded_key", "PUT", AWS1, "/my%20prefix/block%2B0%3D1", "",
         BODY_SHA, "us-east-1"),
    sign("slashes_preserved", "GET", AWS1, "/a/b/c/block_2097152", "",
         EMPTY_SHA, "us-east-1"),
    # What the bdev writes in practice: payload signing opted out.
    sign("unsigned_payload", "PUT", AWS1, "/block_3145728", "", UNSIGNED,
         "us-east-1"),
    # Temporary credentials add x-amz-security-token to the signed headers.
    sign("session_token", "GET", AWS1, "/block_4194304", "", EMPTY_SHA,
         "us-east-1", SESSION_TOKEN),
    # SigV4 is region-scoped; a mismatch is a 301 from S3, not a 403.
    sign("region_us_east_2", "PUT", AWS2, "/clio/block_0", "", UNSIGNED,
         "us-east-2"),
    # S3_ENDPOINT override => path-style: the bucket moves into the URI and
    # the Host header follows. Getting these two out of step is the single
    # easiest way to produce a signature that is wrong only against MinIO.
    sign("path_style_endpoint", "DELETE", "localhost:9000",
         "/clio-bench/clio/block_0", "", EMPTY_SHA, "us-east-2"),
    sign("head_bucket", "HEAD", AWS2, "/", "", EMPTY_SHA, "us-east-2"),
]


def main():
    """Print the C++ initializer list for test_sigv4.cc's kVectors table."""
    for case in CASES:
        (name, method, host, uri, query, payload, region, token,
         signed_headers, cr, sts, authorization) = case
        q = json.dumps  # JSON string escaping is valid C++ for this content
        print("    {")
        print(f"        {q(name)},")
        print(f"        {q(method)}, {q(host)},")
        print(f"        {q(uri)},")
        print(f"        {q(query)}, {q(payload)},")
        print(f"        {q(region)}, {q(token)},")
        print(f"        {q(signed_headers)},")
        print(f"        {q(cr)},")
        print(f"        {q(sts)},")
        print(f"        {q(authorization)},")
        print("    },")


if __name__ == "__main__":
    main()
