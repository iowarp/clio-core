#!/usr/bin/env python3
"""A tiny in-memory S3 stand-in that verifies AWS SigV4, for test_s3_rest.cc.

Python standard library only -- no boto3, no moto, no Docker -- so it runs
anywhere the test does, including Ares, with no credentials and no cloud.

Two things make it worth more than a plain echo server:

  * It VERIFIES the signature, using an implementation derived from the AWS
    SigV4 spec rather than translated from our C++. The frozen vectors in
    test_sigv4.cc pin the signer itself; what they cannot check is how
    S3RestClient *wires* the signer -- the Host header, the canonical URI it
    derives from path-style vs virtual-hosted addressing, the live timestamp.
    Those disagreeing is the classic cause of a signature that is correct in
    isolation and 403s on the wire, so the round trip is checked here.
  * It speaks the exact status codes the transport's semantics depend on:
    404 on a missing object (the sparse zero-fill path) and 200 on bucket HEAD.

Usage (the form CTest uses): start on an ephemeral port, export S3_ENDPOINT and
credentials into the environment, run a child command, exit with its status.

    python3 s3_stub_server.py -- ./clio_run_s3_rest_test
"""

import hashlib
import hmac
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ACCESS_KEY = "AKIDEXAMPLE"
SECRET_KEY = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
REGION = "us-east-2"
BUCKET = "clio-stub-bucket"

# path -> body. Guarded because ThreadingHTTPServer serves each request on its
# own thread, exactly as several runtime workers would.
OBJECTS = {}
LOCK = threading.Lock()

# Reuse proof for test_s3_rest.cc: CONNECTIONS counts accepted TCP sockets,
# REQUESTS counts handled requests. Keep-alive means many REQUESTS ride one
# CONNECTION, so a client that reuses its session shows REQUESTS climbing while
# CONNECTIONS holds. Served back over GET /__stats.
CONNECTIONS = 0
REQUESTS = 0


def _sign(key, msg):
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def signing_key(secret, date, region, service):
    """SigV4 key derivation, per the AWS documentation."""
    k = _sign(("AWS4" + secret).encode("utf-8"), date)
    k = _sign(k, region)
    k = _sign(k, service)
    return _sign(k, "aws4_request")


def verify(method, path, query, headers):
    """Recompute the signature; return None if valid, else a reason string."""
    authz = headers.get("Authorization", "")
    if not authz.startswith("AWS4-HMAC-SHA256 "):
        return "missing or non-SigV4 Authorization header"
    try:
        parts = dict(
            p.strip().split("=", 1) for p in authz[len("AWS4-HMAC-SHA256 "):].split(",")
        )
        credential = parts["Credential"]
        signed_headers = parts["SignedHeaders"]
        provided = parts["Signature"]
        akid, date, region, service, terminator = credential.split("/")
    except (ValueError, KeyError):
        return f"malformed Authorization header: {authz!r}"

    if akid != ACCESS_KEY:
        return f"unknown access key id {akid!r}"
    if terminator != "aws4_request" or service != "s3":
        return f"bad credential scope {credential!r}"

    amz_date = headers.get("x-amz-date", "")
    if not amz_date.startswith(date):
        return f"x-amz-date {amz_date!r} disagrees with scope date {date!r}"

    payload_hash = headers.get("x-amz-content-sha256", "")
    if not payload_hash:
        return "missing x-amz-content-sha256"

    canonical_headers = ""
    for name in signed_headers.split(";"):
        value = headers.get(name)
        if value is None:
            return f"header {name!r} is in SignedHeaders but was not sent"
        canonical_headers += f"{name}:{' '.join(value.split())}\n"

    canonical_request = "\n".join([
        method, path, query, canonical_headers, signed_headers, payload_hash,
    ])
    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256",
        amz_date,
        f"{date}/{region}/{service}/aws4_request",
        hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
    ])
    expected = hmac.new(signing_key(SECRET_KEY, date, region, service),
                        string_to_sign.encode("utf-8"),
                        hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, provided):
        return (f"signature mismatch\n  canonical_request={canonical_request!r}"
                f"\n  expected={expected}\n  provided={provided}")
    return None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        """Runs once per accepted TCP socket -- count it for the reuse proof."""
        super().setup()
        global CONNECTIONS
        with LOCK:
            CONNECTIONS += 1

    def log_message(self, fmt, *args):
        """Quiet by default; set S3_STUB_VERBOSE=1 to trace requests."""
        if os.environ.get("S3_STUB_VERBOSE") == "1":
            sys.stderr.write("[s3-stub] " + (fmt % args) + "\n")

    def _split(self):
        path, _, query = self.path.partition("?")
        return path, query

    def _reject(self, reason):
        body = f"<Error><Code>SignatureDoesNotMatch</Code><Message>{reason}"\
               f"</Message></Error>".encode("utf-8")
        self.log_message("403 %s", reason)
        self.send_response(403)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self, method):
        global REQUESTS
        with LOCK:
            REQUESTS += 1
        path, query = self._split()
        reason = verify(method, path, query, self.headers)
        if reason is not None:
            self._reject(reason)
            return False
        return True

    def _respond(self, status, body=b"", head_only=False):
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body and not head_only:
            self.wfile.write(body)

    def do_HEAD(self):
        if not self._authorized("HEAD"):
            return
        path, _ = self._split()
        # The bucket itself: what EnsureBucket probes.
        if path.rstrip("/") == "/" + BUCKET:
            self._respond(200, head_only=True)
            return
        with LOCK:
            exists = path in OBJECTS
        self._respond(200 if exists else 404, head_only=True)

    def do_PUT(self):
        if not self._authorized("PUT"):
            return
        path, _ = self._split()
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        with LOCK:
            OBJECTS[path] = body
        self._respond(200)

    def do_GET(self):
        path, _ = self._split()
        # Reuse-proof probe. Unauthenticated and uncounted so the test can read
        # the counters without perturbing them.
        if path == "/__stats":
            with LOCK:
                body = f'{{"connections": {CONNECTIONS}, "requests": {REQUESTS}}}'
            self._respond(200, body.encode("utf-8"))
            return
        if not self._authorized("GET"):
            return
        with LOCK:
            body = OBJECTS.get(path)
        if body is None:
            self._respond(404, b"<Error><Code>NoSuchKey</Code></Error>")
            return
        self._respond(200, body)

    def do_DELETE(self):
        if not self._authorized("DELETE"):
            return
        path, _ = self._split()
        with LOCK:
            OBJECTS.pop(path, None)
        self._respond(204)


def main():
    if "--" not in sys.argv:
        sys.exit(__doc__)
    child = sys.argv[sys.argv.index("--") + 1:]
    if not child:
        sys.exit("no child command given after --")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    env = dict(os.environ)
    env.update({
        # Setting S3_ENDPOINT is what selects path-style addressing.
        "S3_ENDPOINT": f"http://127.0.0.1:{port}",
        "AWS_ACCESS_KEY_ID": ACCESS_KEY,
        "AWS_SECRET_ACCESS_KEY": SECRET_KEY,
        "AWS_DEFAULT_REGION": REGION,
        "S3_STUB_BUCKET": BUCKET,
    })
    env.pop("AWS_SESSION_TOKEN", None)  # long-term creds only, so tests are stable
    try:
        return subprocess.call(child, env=env)
    finally:
        server.shutdown()


if __name__ == "__main__":
    sys.exit(main())
