// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).

//! Port of `clio_ctp/encrypt/encrypt.h` + `clio_ctp/encrypt/aes.h` (`ctp::AES`).
//!
//! The C++ class is a thin wrapper over OpenSSL's EVP API: it derives an
//! AES-256 key from a password (`EVP_BytesToKey` + SHA-256), generates a random
//! IV (`RAND_bytes`), and encrypts/decrypts with `EVP_aes_256_cbc` (PKCS#7
//! padding). This crate has no crypto dependency (see the crate's Cargo.toml —
//! `ctp-types` only), so the split is:
//!
//! * **Ported here, as pure logic** — the password KDF (`EVP_BytesToKey` with a
//!   from-scratch SHA-256), CBC chaining, PKCS#7 padding/validation, the
//!   ciphertext-length math, and the `key_`/`iv_`/`salt_` state machine.
//!   The KDF is byte-exact against real OpenSSL output (vectors generated with
//!   `openssl enc -aes-256-cbc -md sha256 -pass pass:... -S ... -P`, OpenSSL
//!   3.5.6, and asserted in the tests below).
//! * **Left to a wrapper crate** — the AES block permutation itself (S-box, key
//!   schedule, 14 rounds). It is abstracted behind [`Aes256BlockCipher`]; this
//!   file deliberately does not hand-roll a cipher. [`UnavailableAes256`] is the
//!   uninhabited placeholder backend, so `Aes<UnavailableAes256>` type-checks
//!   and reports [`CryptoError::CipherUnavailable`] rather than pretending to
//!   encrypt. Everything above the block permutation — the part where a port can
//!   actually get the semantics wrong — is implemented and tested here.
//!
//! C++ → Rust parity map
//! ---------------------
//! | C++ (`clio_ctp/encrypt/aes.h`)              | Rust (this module)                          |
//! |---------------------------------------------|---------------------------------------------|
//! | `ctp::AES`                                  | [`Aes<B>`]                                  |
//! | `AES::key_` (`std::string`, 32 B)           | [`Aes::key`] / [`Aes::set_key`]             |
//! | `AES::iv_` (`std::string`, 16 B)            | [`Aes::iv`] / [`Aes::set_iv`]               |
//! | `AES::salt_` (`std::string`)                | [`Aes::salt`] / [`Aes::salt_pkcs5`]         |
//! | `AES::CreateInitialVector(salt = "")`       | [`Aes::create_initial_vector`]              |
//! | `AES::GenerateKey(password)`                | [`Aes::generate_key`]                       |
//! | `AES::Encrypt(out, out_size, in, in_size)`  | [`Aes::encrypt`] `-> Result<usize>`         |
//! | `AES::Decrypt(out, out_size, in, in_size)`  | [`Aes::decrypt`] `-> Result<usize>`         |
//! | `EVP_BytesToKey(..., count=1, ...)`         | [`evp_bytes_to_key`]                        |
//! | `EVP_sha256()` / `EVP_MD`                   | [`Sha256`] / [`sha256`]                     |
//! | `EVP_aes_256_cbc()`                         | [`Aes256BlockCipher`] + [`cbc_encrypt`] /   |
//! |                                             |   [`cbc_decrypt`]                           |
//! | `EVP_CIPHER_key_length(EVP_aes_256_cbc())`  | [`AES_256_KEY_LEN`] (32)                    |
//! | `EVP_CIPHER_iv_length(EVP_aes_256_cbc())`   | [`AES_256_CBC_IV_LEN`] (16)                 |
//! | `EVP_CIPHER_block_size(...)`                | [`AES_BLOCK_SIZE`] (16)                     |
//! | `PKCS5_SALT_LEN` (openssl/evp.h)            | [`PKCS5_SALT_LEN`] (8)                      |
//! | `RAND_bytes(buf, len)`                      | [`RandomSource::fill_bytes`]                |
//! | `EVP_CIPHER_CTX_new` / `_free`              | (none — RAII; no ctx to leak)               |
//! | `EVP_EncryptUpdate` + `EVP_EncryptFinal_ex` | [`cbc_encrypt`] (single call)               |
//! | `EVP_DecryptUpdate` + `EVP_DecryptFinal_ex` | [`cbc_decrypt`] (single call)               |
//! | `HLOG(kError, "Failed to generate key")`    | (unreachable — see divergence 8)            |
//! | `bool` return                               | [`CryptoError`] via `Result`                |
//! | `#if CTP_ENABLE_ENCRYPT`                    | (always compiled — see divergence 9)        |
//!
//! Semantic divergences (explicit)
//! -------------------------------
//! 1. **The AES block permutation is not implemented here.** [`Aes256BlockCipher`]
//!    defines the backend seam; the default [`UnavailableAes256`] fails every
//!    construction with [`CryptoError::CipherUnavailable`]. A wrapper crate (FFI
//!    to libcrypto, or a vetted `aes` crate) must supply the real backend. No
//!    dependency could be added, since editing Cargo.toml is out of scope.
//! 2. **`Encrypt`'s output layout is fixed.** The C++ writes `EVP_EncryptFinal_ex`
//!    at `output + input_size` and reports `output_size = input_size + final_len`,
//!    but `EVP_EncryptUpdate` only emits `input_size - (input_size % 16)` bytes.
//!    For an unaligned `input_size` the C++ therefore leaves a hole of
//!    uninitialized bytes in `[update_len, input_size)`, writes the final block
//!    past it, and over-reports the length — the ciphertext is unusable. The
//!    two behaviours agree exactly when `input_size % 16 == 0`, which is the only
//!    case the C++ unit test exercises (`test_encrypt.cc` uses 8192 → 8208).
//!    This port always writes the contiguous, correct EVP layout and returns
//!    `(input_size / 16 + 1) * 16` (verified against the OpenSSL CLI for
//!    0/1/10/15/16/17/8192-byte inputs).
//!    Confirmed empirically, not just by reading: compiling the `Encrypt` body
//!    verbatim against OpenSSL 3.5.6 gives, for a 10-byte input, `outl = 0` from
//!    `EVP_EncryptUpdate`, 10 untouched bytes at the front of the buffer,
//!    `output_size = 26` where the true ciphertext is 16, and a round trip that
//!    fails outright.
//! 3. **`Decrypt`'s length is fixed.** The C++ sets `output_size` from
//!    `EVP_DecryptUpdate` only and never adds `EVP_DecryptFinal_ex`'s
//!    `plaintext_len`, so it under-reports by the whole trailing partial block.
//!    It is correct only when the plaintext is block-aligned — again the only
//!    case the C++ test covers. This port returns the true plaintext length.
//!    Also confirmed against the verbatim body + OpenSSL 3.5.6: handed a
//!    *valid* 16-byte ciphertext of a 10-byte plaintext, the C++ reports
//!    `output_size = 0` (update 0 + final 10, final dropped), so a caller doing
//!    `decoded.resize(decoded_size)` — exactly what `test_encrypt.cc` does —
//!    silently keeps nothing. Measured losses: 1→0, 10→0, 15→0, 17→16.
//! 4. **Empty/short `salt_` is defined here, UB in C++.** `EVP_BytesToKey` reads
//!    exactly `PKCS5_SALT_LEN` (8) bytes whenever `salt != NULL`, and
//!    `salt_.c_str()` is never NULL — so the C++ reads up to 8 bytes out of
//!    bounds of a shorter `std::string` (including the default `salt_ == ""`,
//!    the path its own test and docs take), deriving an indeterminate key. This
//!    port zero-pads a short salt and truncates a long one to 8 bytes
//!    ([`Aes::salt_pkcs5`]), making the default case equal to OpenSSL with
//!    `-S 0000000000000000`. The `salt == NULL` (`-nosalt`) case is unreachable
//!    from `ctp::AES`, but [`evp_bytes_to_key`] still models it via `Option`.
//! 5. **`GenerateKey` overwrites `iv_`** — faithfully reproduced.
//!    `EVP_BytesToKey` emits key‖IV, so the derived IV lands in `iv_`. Call order
//!    therefore decides which IV survives, and both orders are ported as-is:
//!    the documented order (`GenerateKey` then `CreateInitialVector`, per the
//!    encryption guide and `test_encrypt.cc`) ends with the *random* IV; the
//!    reverse order ends with the *derived* IV and silently discards the random
//!    one. In C++ the documented order also writes 16 IV bytes into a
//!    still-empty `std::string` (another overflow); here `iv` is a fixed `[u8; 16]`,
//!    so the write is always in bounds.
//! 6. **Out-params → return values.** `bool Encrypt(char*, size_t&, ...)` becomes
//!    `Result<usize, CryptoError>`; the byte count is the `Ok` payload. Buffer
//!    capacity is checked (the C++ checks nothing and overflows the caller's
//!    buffer if it is under-sized), and `&mut [u8]`/`&[u8]` replace
//!    `char* + size_t`. `Decrypt` requires `output.len() >= input.len()`, the
//!    same contract the C++ docs state ("at least input_size bytes").
//! 7. **Unset key/IV is an error, not UB.** A default-constructed C++ `AES` has
//!    empty `key_`/`iv_`, and `Encrypt` hands those to `EVP_EncryptInit_ex`,
//!    which reads 32/16 bytes regardless. This port tracks `key_set`/`iv_set` and
//!    returns [`CryptoError::KeyNotSet`] / [`CryptoError::IvNotSet`].
//! 8. **Error paths differ.** The C++ leaks the `EVP_CIPHER_CTX` on every early
//!    return, ignores `RAND_bytes`'s return code (a silent all-zero IV on RNG
//!    failure), and only `HLOG`s an `EVP_BytesToKey` failure while leaving a zero
//!    key in place. Here there is no ctx to leak, RNG failure propagates
//!    ([`RandomSource::fill_bytes`] returns `Result`), and the KDF is infallible
//!    for the fixed SHA-256/AES-256-CBC pair, so `generate_key` returns `()` like
//!    the C++ — the `HLOG(kError)` branch simply cannot be reached.
//! 9. **No `CTP_ENABLE_ENCRYPT` gate.** The C++ compiles this header to nothing
//!    unless the macro is set. The equivalent would be a cargo feature, which
//!    requires a Cargo.toml edit (out of scope for this file), so the module is
//!    always compiled. Nothing links a cipher yet (divergence 1), so this costs
//!    no dependency.
//! 10. **Key material is not zeroized on drop**, matching the C++ (`std::string`
//!     members are freed, not scrubbed). A `zeroize`-style wipe would need a
//!     dependency or hand-rolled volatile writes; noted for the wrapper crate.
//! 11. **`count = 1`** is preserved from the C++ call. That is a single SHA-256
//!     pass over password‖salt — a deliberately faithful port of a weak KDF, not
//!     an endorsement; OpenSSL itself warns "deprecated key derivation used" and
//!     recommends PBKDF2. Flagged for the wrapper crate rather than silently
//!     "fixed", since changing it would break interop with C++-written data.
//! 12. **64-bit hosts assumed.** `size_t`/`int` mixing in the C++ (`output_size`
//!     is `size_t` but `EVP_*` lengths are `int`, so >2 GiB buffers overflow the
//!     `int`) has no analogue: lengths are `usize` throughout and the ciphertext
//!     length is checked for overflow ([`CryptoError::LengthOverflow`]).

use std::fmt;
use std::marker::PhantomData;

// ---------------------------------------------------------------------------
// Constants (openssl/aes.h, openssl/evp.h equivalents)
// ---------------------------------------------------------------------------

/// AES block size in bytes — `EVP_CIPHER_block_size(EVP_aes_256_cbc())`.
pub const AES_BLOCK_SIZE: usize = 16;

/// AES-256 key length in bytes — `EVP_CIPHER_key_length(EVP_aes_256_cbc())`.
pub const AES_256_KEY_LEN: usize = 32;

/// AES-256-CBC IV length in bytes — `EVP_CIPHER_iv_length(EVP_aes_256_cbc())`,
/// the value the C++ `CreateInitialVector` sizes `iv_` with.
pub const AES_256_CBC_IV_LEN: usize = 16;

/// Salt length consumed by `EVP_BytesToKey` — OpenSSL's `PKCS5_SALT_LEN`.
///
/// `EVP_BytesToKey` hashes exactly this many bytes when `salt != NULL`,
/// regardless of the caller's actual salt length (see divergence 4).
pub const PKCS5_SALT_LEN: usize = 8;

/// Iteration count the C++ `GenerateKey` passes to `EVP_BytesToKey`.
pub const EVP_BYTES_TO_KEY_COUNT: u32 = 1;

/// SHA-256 digest length in bytes.
pub const SHA256_DIGEST_LEN: usize = 32;

// ---------------------------------------------------------------------------
// Errors (C++ returns `bool` / logs; see divergences 6 and 8)
// ---------------------------------------------------------------------------

/// Failure modes of the encryption wrapper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CryptoError {
    /// No AES backend is linked (the default [`UnavailableAes256`]).
    /// Supplying one is wrapper-crate work — see divergence 1.
    CipherUnavailable,
    /// `encrypt`/`decrypt` called before `generate_key`/`set_key`.
    /// The C++ reads an empty `std::string` here instead (divergence 7).
    KeyNotSet,
    /// `encrypt`/`decrypt` called before `create_initial_vector`/`set_iv`.
    IvNotSet,
    /// A key/IV/salt of the wrong length was supplied.
    BadLength {
        /// What the value is for (`"key"`, `"iv"`).
        what: &'static str,
        /// Bytes required.
        expected: usize,
        /// Bytes supplied.
        got: usize,
    },
    /// The output buffer cannot hold the result. The C++ never checks this.
    OutputTooSmall {
        /// Bytes required.
        needed: usize,
        /// Bytes available.
        got: usize,
    },
    /// Ciphertext length is zero or not a multiple of [`AES_BLOCK_SIZE`].
    /// `EVP_DecryptFinal_ex` fails the same way (`false` in the C++).
    InvalidCiphertextLen(usize),
    /// PKCS#7 padding is malformed — wrong key/IV, or tampered ciphertext.
    BadPadding,
    /// The ciphertext length for this plaintext would overflow `usize`.
    LengthOverflow,
    /// The [`RandomSource`] failed. The C++ ignores `RAND_bytes`'s return code.
    RandomSourceFailed,
}

impl fmt::Display for CryptoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CipherUnavailable => {
                write!(f, "no AES-256 backend is linked (wrapper-crate work)")
            }
            Self::KeyNotSet => write!(f, "key not set: call generate_key() or set_key() first"),
            Self::IvNotSet => write!(
                f,
                "iv not set: call create_initial_vector() or set_iv() first"
            ),
            Self::BadLength {
                what,
                expected,
                got,
            } => write!(f, "{what} must be {expected} bytes, got {got}"),
            Self::OutputTooSmall { needed, got } => {
                write!(f, "output buffer too small: need {needed} bytes, got {got}")
            }
            Self::InvalidCiphertextLen(n) => write!(
                f,
                "ciphertext length {n} is not a non-zero multiple of {AES_BLOCK_SIZE}"
            ),
            Self::BadPadding => write!(f, "invalid PKCS#7 padding (wrong key/iv or tampered data)"),
            Self::LengthOverflow => write!(f, "ciphertext length would overflow usize"),
            Self::RandomSourceFailed => write!(f, "random source failed"),
        }
    }
}

impl std::error::Error for CryptoError {}

/// Convenience alias for this module's results.
pub type Result<T> = std::result::Result<T, CryptoError>;

// ---------------------------------------------------------------------------
// SHA-256 (EVP_sha256 equivalent) — pure logic, FIPS 180-4
// ---------------------------------------------------------------------------

const SHA256_INIT: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

#[rustfmt::skip]
const SHA256_K: [u32; 64] = [
    0x428a_2f98, 0x7137_4491, 0xb5c0_fbcf, 0xe9b5_dba5, 0x3956_c25b, 0x59f1_11f1, 0x923f_82a4, 0xab1c_5ed5,
    0xd807_aa98, 0x1283_5b01, 0x2431_85be, 0x550c_7dc3, 0x72be_5d74, 0x80de_b1fe, 0x9bdc_06a7, 0xc19b_f174,
    0xe49b_69c1, 0xefbe_4786, 0x0fc1_9dc6, 0x240c_a1cc, 0x2de9_2c6f, 0x4a74_84aa, 0x5cb0_a9dc, 0x76f9_88da,
    0x983e_5152, 0xa831_c66d, 0xb003_27c8, 0xbf59_7fc7, 0xc6e0_0bf3, 0xd5a7_9147, 0x06ca_6351, 0x1429_2967,
    0x27b7_0a85, 0x2e1b_2138, 0x4d2c_6dfc, 0x5338_0d13, 0x650a_7354, 0x766a_0abb, 0x81c2_c92e, 0x9272_2c85,
    0xa2bf_e8a1, 0xa81a_664b, 0xc24b_8b70, 0xc76c_51a3, 0xd192_e819, 0xd699_0624, 0xf40e_3585, 0x106a_a070,
    0x19a4_c116, 0x1e37_6c08, 0x2748_774c, 0x34b0_bcb5, 0x391c_0cb3, 0x4ed8_aa4a, 0x5b9c_ca4f, 0x682e_6ff3,
    0x748f_82ee, 0x78a5_636f, 0x84c8_7814, 0x8cc7_0208, 0x90be_fffa, 0xa450_6ceb, 0xbef9_a3f7, 0xc671_78f2,
];

fn sha256_compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for (word, chunk) in w.iter_mut().zip(block.chunks_exact(4)) {
        *word = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    for i in 16..64 {
        let x = w[i - 15];
        let y = w[i - 2];
        let s0 = x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3);
        let s1 = y.rotate_right(17) ^ y.rotate_right(19) ^ (y >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for (k, wi) in SHA256_K.iter().zip(w.iter()) {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(*k)
            .wrapping_add(*wi);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }

    for (s, v) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *s = s.wrapping_add(v);
    }
}

/// Incremental SHA-256 — the `EVP_MD_CTX` + `EVP_sha256()` pair the C++
/// `GenerateKey` passes to `EVP_BytesToKey`.
///
/// Implemented from scratch (FIPS 180-4) because the crate may not take a
/// dependency; validated against the NIST vectors and the OpenSSL CLI in the
/// tests. `update`/`finalize` mirror `EVP_DigestUpdate`/`EVP_DigestFinal_ex`.
#[derive(Debug, Clone)]
pub struct Sha256 {
    state: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_len: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    /// `EVP_DigestInit_ex(ctx, EVP_sha256(), NULL)`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: SHA256_INIT,
            buf: [0u8; 64],
            buf_len: 0,
            total_len: 0,
        }
    }

    /// `EVP_DigestUpdate` — absorb more input. Splitting the input across calls
    /// never changes the digest.
    pub fn update(&mut self, data: &[u8]) {
        let mut data = data;
        self.total_len = self.total_len.wrapping_add(data.len() as u64);

        // Top up a partially filled block first.
        if self.buf_len > 0 {
            let take = core::cmp::min(64 - self.buf_len, data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                sha256_compress(&mut self.state, &block);
                self.buf_len = 0;
            }
        }
        if data.is_empty() {
            return;
        }
        // Invariant: buf_len == 0 here, so whole blocks stream straight through.
        debug_assert_eq!(self.buf_len, 0);

        let mut chunks = data.chunks_exact(64);
        for chunk in &mut chunks {
            let mut block = [0u8; 64];
            block.copy_from_slice(chunk);
            sha256_compress(&mut self.state, &block);
        }
        let rem = chunks.remainder();
        self.buf[..rem.len()].copy_from_slice(rem);
        self.buf_len = rem.len();
    }

    /// `EVP_DigestFinal_ex` — apply the padding and emit the digest.
    #[must_use]
    pub fn finalize(mut self) -> [u8; SHA256_DIGEST_LEN] {
        let bit_len = self.total_len.wrapping_mul(8);

        // buf_len < 64 always (a full block is compressed immediately).
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;
        if self.buf_len > 56 {
            for b in self.buf[self.buf_len..].iter_mut() {
                *b = 0;
            }
            let block = self.buf;
            sha256_compress(&mut self.state, &block);
            self.buf = [0u8; 64];
            self.buf_len = 0;
        }
        for b in self.buf[self.buf_len..56].iter_mut() {
            *b = 0;
        }
        self.buf[56..].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buf;
        sha256_compress(&mut self.state, &block);

        let mut out = [0u8; SHA256_DIGEST_LEN];
        for (chunk, h) in out.chunks_exact_mut(4).zip(self.state.iter()) {
            chunk.copy_from_slice(&h.to_be_bytes());
        }
        out
    }
}

/// One-shot SHA-256 (`EVP_Digest(..., EVP_sha256(), ...)`).
#[must_use]
pub fn sha256(data: &[u8]) -> [u8; SHA256_DIGEST_LEN] {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize()
}

// ---------------------------------------------------------------------------
// EVP_BytesToKey (the KDF behind GenerateKey) — pure logic
// ---------------------------------------------------------------------------

/// Port of OpenSSL's `EVP_BytesToKey` specialised to the digest the C++ uses
/// (`EVP_sha256()`).
///
/// Reproduces the OpenSSL loop exactly:
///
/// ```text
/// D_1     = SHA256(password ‖ salt)
/// D_i     = SHA256(D_{i-1} ‖ password ‖ salt)          for i > 1
/// D_i    := SHA256^(count-1)(D_i)                      (extra iterations)
/// key‖iv  = D_1 ‖ D_2 ‖ …                              (truncated to fit)
/// ```
///
/// * `salt` is `Option<&[u8; PKCS5_SALT_LEN]>`: `Some` hashes exactly 8 bytes
///   (what `ctp::AES` always does — divergence 4), `None` models OpenSSL's
///   `salt == NULL` / `openssl enc -nosalt`, which hashes no salt at all.
/// * `count == 0` behaves like `count == 1`, matching the C++ `for (i = 1; i <
///   (unsigned)count; i++)` loop, which runs zero extra iterations either way.
/// * `key_out` is filled before `iv_out`, both to their full length; either may
///   be empty.
///
/// Byte-exact against `openssl enc -aes-256-cbc -md sha256 -P` (see tests).
pub fn evp_bytes_to_key(
    password: &[u8],
    salt: Option<&[u8; PKCS5_SALT_LEN]>,
    count: u32,
    key_out: &mut [u8],
    iv_out: &mut [u8],
) {
    let mut md_buf = [0u8; SHA256_DIGEST_LEN];
    let mut addmd = false;
    let mut key_done = 0usize;
    let mut iv_done = 0usize;

    loop {
        let mut ctx = Sha256::new();
        if addmd {
            ctx.update(&md_buf);
        }
        addmd = true;
        ctx.update(password);
        if let Some(s) = salt {
            ctx.update(s);
        }
        md_buf = ctx.finalize();

        // `count - 1` further passes over the digest alone.
        for _ in 1..count {
            md_buf = sha256(&md_buf);
        }

        // Spill the digest into key first, then iv (OpenSSL's byte-at-a-time
        // copy, expressed as slices).
        let mut avail = &md_buf[..];
        if key_done < key_out.len() {
            let take = core::cmp::min(key_out.len() - key_done, avail.len());
            key_out[key_done..key_done + take].copy_from_slice(&avail[..take]);
            key_done += take;
            avail = &avail[take..];
        }
        if iv_done < iv_out.len() && !avail.is_empty() {
            let take = core::cmp::min(iv_out.len() - iv_done, avail.len());
            iv_out[iv_done..iv_done + take].copy_from_slice(&avail[..take]);
            iv_done += take;
        }
        if key_done == key_out.len() && iv_done == iv_out.len() {
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Cipher + RNG seams (the OpenSSL-provided pieces)
// ---------------------------------------------------------------------------

/// The AES-256 block permutation — the one piece this crate cannot supply.
///
/// This is the seam where a wrapper crate plugs in libcrypto (or a vetted pure
/// Rust `aes`); [`cbc_encrypt`]/[`cbc_decrypt`] build the CBC mode and PKCS#7
/// padding of `EVP_aes_256_cbc()` on top of it. Implementations operate on a
/// single [`AES_BLOCK_SIZE`] block in place and must be exact inverses.
pub trait Aes256BlockCipher: Sized {
    /// Build a cipher from a 256-bit key (the key schedule).
    ///
    /// # Errors
    /// [`CryptoError::CipherUnavailable`] when no backend is linked.
    fn new(key: &[u8; AES_256_KEY_LEN]) -> Result<Self>;

    /// Encrypt one block in place.
    fn encrypt_block(&self, block: &mut [u8; AES_BLOCK_SIZE]);

    /// Decrypt one block in place (exact inverse of [`Self::encrypt_block`]).
    fn decrypt_block(&self, block: &mut [u8; AES_BLOCK_SIZE]);
}

/// The placeholder backend: no cipher is linked into this crate.
///
/// Uninhabited on purpose — [`Aes256BlockCipher::new`] always fails with
/// [`CryptoError::CipherUnavailable`], so a value can never exist and the block
/// methods are statically unreachable. It keeps `Aes<UnavailableAes256>` (see
/// [`AesDefault`]) usable for key/IV derivation, which is fully implemented,
/// while making the missing cipher a loud, typed error instead of a silent
/// insecure fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnavailableAes256 {}

impl Aes256BlockCipher for UnavailableAes256 {
    fn new(_key: &[u8; AES_256_KEY_LEN]) -> Result<Self> {
        Err(CryptoError::CipherUnavailable)
    }

    fn encrypt_block(&self, _block: &mut [u8; AES_BLOCK_SIZE]) {
        match *self {}
    }

    fn decrypt_block(&self, _block: &mut [u8; AES_BLOCK_SIZE]) {
        match *self {}
    }
}

/// `RAND_bytes` equivalent: a source of cryptographically secure random bytes.
///
/// std has no CSPRNG, so — like the cipher — the real implementation belongs to
/// a wrapper crate (libcrypto's `RAND_bytes`, `getrandom`, `BCryptGenRandom`).
/// Unlike the C++, failure is reported rather than silently yielding a zero IV.
pub trait RandomSource {
    /// Fill `dest` with random bytes.
    ///
    /// # Errors
    /// [`CryptoError::RandomSourceFailed`] if the underlying source fails.
    fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<()>;
}

// ---------------------------------------------------------------------------
// CBC mode + PKCS#7 padding (what EVP does around the block cipher) — pure
// ---------------------------------------------------------------------------

/// Ciphertext length AES-256-CBC + PKCS#7 produces for `plaintext_len`:
/// `(plaintext_len / 16 + 1) * 16`. Padding is always added, so a block-aligned
/// plaintext grows by a whole block (0 → 16, 16 → 32, 8192 → 8208 — matching
/// the OpenSSL CLI).
///
/// # Errors
/// [`CryptoError::LengthOverflow`] if the result would exceed `usize::MAX`.
pub fn pkcs7_ciphertext_len(plaintext_len: usize) -> Result<usize> {
    let full = plaintext_len - (plaintext_len % AES_BLOCK_SIZE);
    full.checked_add(AES_BLOCK_SIZE)
        .ok_or(CryptoError::LengthOverflow)
}

fn xor_block(dst: &mut [u8; AES_BLOCK_SIZE], src: &[u8; AES_BLOCK_SIZE]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d ^= *s;
    }
}

fn to_block(bytes: &[u8]) -> [u8; AES_BLOCK_SIZE] {
    let mut b = [0u8; AES_BLOCK_SIZE];
    b.copy_from_slice(bytes);
    b
}

/// AES-256-CBC encryption with PKCS#7 padding — the mode logic of
/// `EVP_EncryptUpdate` + `EVP_EncryptFinal_ex`, without the C++'s output-offset
/// bug (divergence 2). Returns the number of ciphertext bytes written.
///
/// # Errors
/// [`CryptoError::OutputTooSmall`] if `output` is shorter than
/// [`pkcs7_ciphertext_len`]; [`CryptoError::LengthOverflow`] on length overflow.
pub fn cbc_encrypt<B: Aes256BlockCipher>(
    cipher: &B,
    iv: &[u8; AES_256_CBC_IV_LEN],
    input: &[u8],
    output: &mut [u8],
) -> Result<usize> {
    let needed = pkcs7_ciphertext_len(input.len())?;
    if output.len() < needed {
        return Err(CryptoError::OutputTooSmall {
            needed,
            got: output.len(),
        });
    }

    let mut prev = *iv;
    let mut written = 0usize;
    let mut chunks = input.chunks_exact(AES_BLOCK_SIZE);
    for chunk in &mut chunks {
        let mut block = to_block(chunk);
        xor_block(&mut block, &prev);
        cipher.encrypt_block(&mut block);
        output[written..written + AES_BLOCK_SIZE].copy_from_slice(&block);
        prev = block;
        written += AES_BLOCK_SIZE;
    }

    // The always-present padded final block (PKCS#7: pad value == pad length,
    // a full block when the input is already aligned).
    let rem = chunks.remainder();
    let pad = AES_BLOCK_SIZE - rem.len();
    let mut block = [pad as u8; AES_BLOCK_SIZE];
    block[..rem.len()].copy_from_slice(rem);
    xor_block(&mut block, &prev);
    cipher.encrypt_block(&mut block);
    output[written..written + AES_BLOCK_SIZE].copy_from_slice(&block);
    written += AES_BLOCK_SIZE;

    debug_assert_eq!(written, needed);
    Ok(written)
}

/// AES-256-CBC decryption with PKCS#7 validation — the mode logic of
/// `EVP_DecryptUpdate` + `EVP_DecryptFinal_ex`, reporting the *full* plaintext
/// length (divergence 3). Returns the number of plaintext bytes written.
///
/// `output` must be at least `input.len()` bytes, the contract the C++ docs
/// state; the plaintext itself is always shorter.
///
/// # Errors
/// [`CryptoError::InvalidCiphertextLen`] if `input` is empty or not block
/// aligned; [`CryptoError::OutputTooSmall`]; [`CryptoError::BadPadding`] if the
/// PKCS#7 trailer is malformed (wrong key/IV or tampered data) — all cases where
/// the C++ `Decrypt` returns `false`.
pub fn cbc_decrypt<B: Aes256BlockCipher>(
    cipher: &B,
    iv: &[u8; AES_256_CBC_IV_LEN],
    input: &[u8],
    output: &mut [u8],
) -> Result<usize> {
    if input.is_empty() || !input.len().is_multiple_of(AES_BLOCK_SIZE) {
        return Err(CryptoError::InvalidCiphertextLen(input.len()));
    }
    if output.len() < input.len() {
        return Err(CryptoError::OutputTooSmall {
            needed: input.len(),
            got: output.len(),
        });
    }

    let mut prev = *iv;
    let mut written = 0usize;
    let bulk = input.len() - AES_BLOCK_SIZE;
    for chunk in input[..bulk].chunks_exact(AES_BLOCK_SIZE) {
        let cipher_block = to_block(chunk);
        let mut block = cipher_block;
        cipher.decrypt_block(&mut block);
        xor_block(&mut block, &prev);
        output[written..written + AES_BLOCK_SIZE].copy_from_slice(&block);
        prev = cipher_block;
        written += AES_BLOCK_SIZE;
    }

    // Final block: decrypt, then strip and validate the PKCS#7 trailer.
    let mut last = to_block(&input[bulk..]);
    cipher.decrypt_block(&mut last);
    xor_block(&mut last, &prev);
    let keep = pkcs7_unpad_len(&last)?;
    output[written..written + keep].copy_from_slice(&last[..keep]);
    written += keep;

    Ok(written)
}

/// Validate the PKCS#7 trailer of a decrypted final block and return how many
/// of its bytes are plaintext.
///
/// Mirrors `EVP_DecryptFinal_ex`'s check: the pad length must be in `1..=16` and
/// every padding byte must equal it.
///
/// # Errors
/// [`CryptoError::BadPadding`] if the trailer is malformed.
pub fn pkcs7_unpad_len(block: &[u8; AES_BLOCK_SIZE]) -> Result<usize> {
    let pad = block[AES_BLOCK_SIZE - 1] as usize;
    if pad == 0 || pad > AES_BLOCK_SIZE {
        return Err(CryptoError::BadPadding);
    }
    if block[AES_BLOCK_SIZE - pad..]
        .iter()
        .any(|&b| b as usize != pad)
    {
        return Err(CryptoError::BadPadding);
    }
    Ok(AES_BLOCK_SIZE - pad)
}

// ---------------------------------------------------------------------------
// ctp::AES
// ---------------------------------------------------------------------------

/// Port of `ctp::AES` (`clio_ctp/encrypt/aes.h`).
///
/// Generic over the block-cipher backend so the (unported) AES permutation stays
/// a wrapper-crate concern; everything else — key derivation, IV handling, CBC,
/// padding — is implemented here. [`AesDefault`] is the no-backend instantiation.
///
/// The C++ exposes `key_`/`iv_`/`salt_` as public `std::string`s; this port keeps
/// them private behind accessors so the length invariants (32/16 bytes) hold by
/// construction, and tracks whether they have been set (divergence 7).
///
/// # Call order
/// `GenerateKey` writes *both* key and IV (`EVP_BytesToKey` emits key‖IV), so
/// order matters and is preserved exactly (divergence 5):
///
/// ```text
/// generate_key(pw); create_initial_vector(salt, rng);  // random IV wins  (documented order)
/// create_initial_vector(salt, rng); generate_key(pw);  // derived IV wins, random IV discarded
/// ```
///
/// Only the second order actually feeds the salt into the key derivation.
#[derive(Debug, Clone)]
pub struct Aes<B: Aes256BlockCipher> {
    key: [u8; AES_256_KEY_LEN],
    iv: [u8; AES_256_CBC_IV_LEN],
    salt: Vec<u8>,
    key_set: bool,
    iv_set: bool,
    // fn() -> B keeps Aes<B> Send/Sync/Clone regardless of B.
    _cipher: PhantomData<fn() -> B>,
}

/// `Aes` with no cipher backend linked — key/IV derivation works, encryption
/// reports [`CryptoError::CipherUnavailable`]. See divergence 1.
pub type AesDefault = Aes<UnavailableAes256>;

impl<B: Aes256BlockCipher> Default for Aes<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Aes256BlockCipher> Aes<B> {
    /// A default-constructed `ctp::AES`: no key, no IV, empty salt.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            key: [0u8; AES_256_KEY_LEN],
            iv: [0u8; AES_256_CBC_IV_LEN],
            salt: Vec::new(),
            key_set: false,
            iv_set: false,
            _cipher: PhantomData,
        }
    }

    /// The derived key (`key_`).
    #[must_use]
    pub const fn key(&self) -> &[u8; AES_256_KEY_LEN] {
        &self.key
    }

    /// The initialization vector (`iv_`).
    #[must_use]
    pub const fn iv(&self) -> &[u8; AES_256_CBC_IV_LEN] {
        &self.iv
    }

    /// The stored salt (`salt_`), as supplied — not yet padded to
    /// [`PKCS5_SALT_LEN`].
    #[must_use]
    pub fn salt(&self) -> &[u8] {
        &self.salt
    }

    /// True once `generate_key`/`set_key` has run.
    #[must_use]
    pub const fn has_key(&self) -> bool {
        self.key_set
    }

    /// True once `create_initial_vector`/`generate_key`/`set_iv` has run.
    #[must_use]
    pub const fn has_iv(&self) -> bool {
        self.iv_set
    }

    /// Install a key directly (the C++ writes the public `key_` member).
    ///
    /// # Errors
    /// [`CryptoError::BadLength`] unless `key` is [`AES_256_KEY_LEN`] bytes.
    pub fn set_key(&mut self, key: &[u8]) -> Result<()> {
        if key.len() != AES_256_KEY_LEN {
            return Err(CryptoError::BadLength {
                what: "key",
                expected: AES_256_KEY_LEN,
                got: key.len(),
            });
        }
        self.key.copy_from_slice(key);
        self.key_set = true;
        Ok(())
    }

    /// Install an IV directly (the C++ writes the public `iv_` member); this is
    /// how a receiver adopts the sender's IV.
    ///
    /// # Errors
    /// [`CryptoError::BadLength`] unless `iv` is [`AES_256_CBC_IV_LEN`] bytes.
    pub fn set_iv(&mut self, iv: &[u8]) -> Result<()> {
        if iv.len() != AES_256_CBC_IV_LEN {
            return Err(CryptoError::BadLength {
                what: "iv",
                expected: AES_256_CBC_IV_LEN,
                got: iv.len(),
            });
        }
        self.iv.copy_from_slice(iv);
        self.iv_set = true;
        Ok(())
    }

    /// The salt as `EVP_BytesToKey` consumes it: exactly [`PKCS5_SALT_LEN`]
    /// bytes, zero-padded if `salt_` is shorter, truncated if longer.
    ///
    /// The C++ instead reads 8 bytes from `salt_.c_str()` unconditionally,
    /// running off the end of any shorter string — including the default empty
    /// one (divergence 4). Zero-padding makes the default deterministic and
    /// equal to `openssl enc -S 0000000000000000`.
    #[must_use]
    pub fn salt_pkcs5(&self) -> [u8; PKCS5_SALT_LEN] {
        let mut out = [0u8; PKCS5_SALT_LEN];
        let take = core::cmp::min(PKCS5_SALT_LEN, self.salt.len());
        out[..take].copy_from_slice(&self.salt[..take]);
        out
    }

    /// `void CreateInitialVector(const std::string &salt = "")`.
    ///
    /// Stores `salt` (for a later [`Self::generate_key`]) and fills `iv_` with
    /// [`AES_256_CBC_IV_LEN`] random bytes. Pass `&[]` for the C++ default
    /// argument. Unlike the C++, RNG failure is reported and the RNG is an
    /// explicit parameter rather than the ambient `RAND_bytes`.
    ///
    /// # Errors
    /// [`CryptoError::RandomSourceFailed`] if `rng` fails; `iv_` is then left
    /// unset rather than silently zeroed.
    pub fn create_initial_vector<R: RandomSource + ?Sized>(
        &mut self,
        salt: &[u8],
        rng: &mut R,
    ) -> Result<()> {
        self.salt.clear();
        self.salt.extend_from_slice(salt);
        let mut iv = [0u8; AES_256_CBC_IV_LEN];
        rng.fill_bytes(&mut iv)?;
        self.iv = iv;
        self.iv_set = true;
        Ok(())
    }

    /// `void GenerateKey(const std::string &password)`.
    ///
    /// `EVP_BytesToKey(EVP_aes_256_cbc(), EVP_sha256(), salt_, password, 1, key_, iv_)`.
    /// Derives `key_` **and overwrites `iv_`** with the KDF's IV output, exactly
    /// as the C++ does (divergence 5). Infallible for this digest/cipher pair, so
    /// the C++'s `HLOG(kError, ...)` branch has no analogue (divergence 8).
    pub fn generate_key(&mut self, password: &[u8]) {
        let salt = self.salt_pkcs5();
        let mut key = [0u8; AES_256_KEY_LEN];
        let mut iv = [0u8; AES_256_CBC_IV_LEN];
        evp_bytes_to_key(
            password,
            Some(&salt),
            EVP_BYTES_TO_KEY_COUNT,
            &mut key,
            &mut iv,
        );
        self.key = key;
        self.iv = iv;
        self.key_set = true;
        self.iv_set = true;
    }

    /// Bytes [`Self::encrypt`] needs in `output` for `plaintext_len` input.
    ///
    /// # Errors
    /// [`CryptoError::LengthOverflow`] on overflow.
    pub fn encrypt_output_len(plaintext_len: usize) -> Result<usize> {
        pkcs7_ciphertext_len(plaintext_len)
    }

    /// `bool Encrypt(char *output, size_t &output_size, char *input, size_t input_size)`.
    ///
    /// Returns the ciphertext length instead of writing an out-param
    /// (divergence 6). The layout is the correct EVP one, which matches the C++
    /// exactly when `input.len() % 16 == 0` (divergence 2).
    ///
    /// # Errors
    /// [`CryptoError::KeyNotSet`] / [`CryptoError::IvNotSet`] if the key/IV were
    /// never derived (UB in the C++); [`CryptoError::OutputTooSmall`] if `output`
    /// is under-sized (a buffer overflow in the C++);
    /// [`CryptoError::CipherUnavailable`] with no backend linked.
    pub fn encrypt(&self, output: &mut [u8], input: &[u8]) -> Result<usize> {
        let cipher = self.cipher()?;
        cbc_encrypt(&cipher, &self.iv, input, output)
    }

    /// `bool Decrypt(char *output, size_t &output_size, char *input, size_t input_size)`.
    ///
    /// Returns the *complete* plaintext length, including the final block the
    /// C++ forgets to count (divergence 3).
    ///
    /// # Errors
    /// [`CryptoError::KeyNotSet`] / [`CryptoError::IvNotSet`];
    /// [`CryptoError::InvalidCiphertextLen`]; [`CryptoError::OutputTooSmall`];
    /// [`CryptoError::BadPadding`]; [`CryptoError::CipherUnavailable`]. Each of
    /// these is a `false` return from the C++ `Decrypt` (or UB).
    pub fn decrypt(&self, output: &mut [u8], input: &[u8]) -> Result<usize> {
        let cipher = self.cipher()?;
        cbc_decrypt(&cipher, &self.iv, input, output)
    }

    /// `EVP_{Encrypt,Decrypt}Init_ex` equivalent: check state, build the cipher.
    fn cipher(&self) -> Result<B> {
        if !self.key_set {
            return Err(CryptoError::KeyNotSet);
        }
        if !self.iv_set {
            return Err(CryptoError::IvNotSet);
        }
        B::new(&self.key)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers ------------------------------------------------------------

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn unhex(s: &str) -> Vec<u8> {
        assert!(s.len().is_multiple_of(2), "odd hex string");
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).expect("valid hex"))
            .collect()
    }

    /// A stand-in for the real AES permutation, so the ported CBC/padding logic
    /// can be tested end-to-end. `E(x) = reverse(x) ^ k`, `D(y) = reverse(y ^ k)`
    /// — an exact inverse pair, key dependent, and emphatically NOT a cipher.
    struct FakeBlockCipher {
        k: [u8; AES_BLOCK_SIZE],
    }

    impl Aes256BlockCipher for FakeBlockCipher {
        fn new(key: &[u8; AES_256_KEY_LEN]) -> Result<Self> {
            let mut k = [0u8; AES_BLOCK_SIZE];
            for (i, slot) in k.iter_mut().enumerate() {
                *slot = key[i] ^ key[i + AES_BLOCK_SIZE];
            }
            Ok(Self { k })
        }

        fn encrypt_block(&self, block: &mut [u8; AES_BLOCK_SIZE]) {
            block.reverse();
            xor_block(block, &self.k);
        }

        fn decrypt_block(&self, block: &mut [u8; AES_BLOCK_SIZE]) {
            xor_block(block, &self.k);
            block.reverse();
        }
    }

    /// Deterministic stand-in for `RAND_bytes` (a real CSPRNG is wrapper work).
    struct CounterRng {
        next: u8,
        fail: bool,
    }

    impl CounterRng {
        fn new(start: u8) -> Self {
            Self {
                next: start,
                fail: false,
            }
        }
        fn failing() -> Self {
            Self {
                next: 0,
                fail: true,
            }
        }
    }

    impl RandomSource for CounterRng {
        fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<()> {
            if self.fail {
                return Err(CryptoError::RandomSourceFailed);
            }
            for b in dest.iter_mut() {
                *b = self.next;
                self.next = self.next.wrapping_add(1);
            }
            Ok(())
        }
    }

    fn keyed_aes() -> Aes<FakeBlockCipher> {
        let mut aes = Aes::<FakeBlockCipher>::new();
        aes.generate_key(b"passwd");
        aes
    }

    // -- SHA-256 ------------------------------------------------------------

    #[test]
    fn sha256_nist_vectors() {
        // FIPS 180-4 / NIST examples, cross-checked with `openssl dgst -sha256`.
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hex(&sha256(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    #[rustfmt::skip] // keep the digest table readable, one vector per line
    fn sha256_length_boundaries() {
        // The padding boundaries: 55 (fits), 56 (forces a second block), 63, 64
        // (exact block), 65, and a multi-block input.
        let expect = [
            (55usize, "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"),
            (56, "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"),
            (63, "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34"),
            (64, "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"),
            (65, "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0"),
            (1000, "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3"),
        ];
        for (n, want) in expect {
            let data = vec![b'a'; n];
            assert_eq!(hex(&sha256(&data)), want, "sha256 of {n} 'a' bytes");
        }
    }

    #[test]
    fn sha256_incremental_matches_one_shot() {
        // Splitting the input at any point must not change the digest — the
        // property EVP_BytesToKey relies on when it feeds password then salt.
        let data: Vec<u8> = (0..=255u8).cycle().take(700).collect();
        let want = sha256(&data);
        for split in [0usize, 1, 63, 64, 65, 127, 128, 129, 699, 700] {
            let mut h = Sha256::new();
            h.update(&data[..split]);
            h.update(&data[split..]);
            assert_eq!(h.finalize(), want, "split at {split}");
        }
        // Many tiny updates across block boundaries.
        let mut h = Sha256::new();
        for byte in &data {
            h.update(std::slice::from_ref(byte));
        }
        assert_eq!(h.finalize(), want, "byte-at-a-time");
        // Empty updates are no-ops.
        let mut h = Sha256::new();
        h.update(&[]);
        h.update(&data);
        h.update(&[]);
        assert_eq!(h.finalize(), want, "empty updates");
    }

    // -- EVP_BytesToKey -----------------------------------------------------

    // Vectors below are real OpenSSL 3.5.6 output:
    //   openssl enc -aes-256-cbc -md sha256 -pass pass:<pw> -S <salt> -P
    // i.e. EVP_BytesToKey(EVP_aes_256_cbc(), EVP_sha256(), salt, pw, 1, k, iv)
    // — exactly the call the C++ GenerateKey makes.
    //
    // The zero-salt/"passwd" pair was additionally cross-checked by compiling
    // GenerateKey's EVP_BytesToKey call verbatim against libcrypto, which emits
    // the same key/iv this port derives — so the KDF port is byte-exact against
    // the real C++ path, not merely self-consistent.

    #[test]
    fn evp_bytes_to_key_matches_openssl_zero_salt() {
        // -S 0000000000000000 == this port's empty-salt_ modelling (divergence 4).
        let mut key = [0u8; AES_256_KEY_LEN];
        let mut iv = [0u8; AES_256_CBC_IV_LEN];
        evp_bytes_to_key(b"passwd", Some(&[0u8; 8]), 1, &mut key, &mut iv);
        assert_eq!(
            hex(&key),
            "0f151149ec2da0732ef63eec89d45b413f07eb9331a060c27e9c6816a69bbe46"
        );
        assert_eq!(hex(&iv), "1ed2bfe61f5ffe027f95143ca4f58df8");
    }

    #[test]
    fn evp_bytes_to_key_matches_openssl_nonzero_salt() {
        let salt = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut key = [0u8; AES_256_KEY_LEN];
        let mut iv = [0u8; AES_256_CBC_IV_LEN];
        evp_bytes_to_key(b"passwd", Some(&salt), 1, &mut key, &mut iv);
        assert_eq!(
            hex(&key),
            "4af221335492ba211823e8dc89e76697f0f934836e95a155e838fc783091d9bc"
        );
        assert_eq!(hex(&iv), "17600c44c1385a55b1135509d4efb143");
    }

    #[test]
    fn evp_bytes_to_key_matches_openssl_no_salt_and_empty_password() {
        // salt = None models `openssl enc -nosalt` (unreachable from ctp::AES,
        // but part of EVP_BytesToKey's contract).
        let mut key = [0u8; AES_256_KEY_LEN];
        let mut iv = [0u8; AES_256_CBC_IV_LEN];
        evp_bytes_to_key(b"passwd", None, 1, &mut key, &mut iv);
        assert_eq!(
            hex(&key),
            "0d6be69b264717f2dd33652e212b173104b4a647b7c11ae72e9885f11cd312fb"
        );
        assert_eq!(hex(&iv), "bfac3a948f96dd193287e145d85dc4c3");

        // Empty password, zero salt.
        let mut key = [0u8; AES_256_KEY_LEN];
        let mut iv = [0u8; AES_256_CBC_IV_LEN];
        evp_bytes_to_key(b"", Some(&[0u8; 8]), 1, &mut key, &mut iv);
        assert_eq!(
            hex(&key),
            "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc"
        );
        assert_eq!(hex(&iv), "6529637920af0dab831d04ff378fa103");
    }

    #[test]
    fn evp_bytes_to_key_structure_and_counts() {
        // key = D1, iv = D2[..16] where D1 = H(pw‖salt), D2 = H(D1‖pw‖salt).
        let salt = [9u8; 8];
        let pw = b"hunter2";
        let mut d1_in = pw.to_vec();
        d1_in.extend_from_slice(&salt);
        let d1 = sha256(&d1_in);
        let mut d2_in = d1.to_vec();
        d2_in.extend_from_slice(pw);
        d2_in.extend_from_slice(&salt);
        let d2 = sha256(&d2_in);

        let mut key = [0u8; AES_256_KEY_LEN];
        let mut iv = [0u8; AES_256_CBC_IV_LEN];
        evp_bytes_to_key(pw, Some(&salt), 1, &mut key, &mut iv);
        assert_eq!(key, d1);
        assert_eq!(iv, d2[..AES_256_CBC_IV_LEN]);

        // count = 0 behaves like count = 1 (the C++ `for (i=1; i<count; i++)`).
        let mut key0 = [0u8; AES_256_KEY_LEN];
        let mut iv0 = [0u8; AES_256_CBC_IV_LEN];
        evp_bytes_to_key(pw, Some(&salt), 0, &mut key0, &mut iv0);
        assert_eq!((key0, iv0), (key, iv));

        // count = 2 hashes each D once more, so it must differ.
        let mut key2 = [0u8; AES_256_KEY_LEN];
        let mut iv2 = [0u8; AES_256_CBC_IV_LEN];
        evp_bytes_to_key(pw, Some(&salt), 2, &mut key2, &mut iv2);
        assert_eq!(key2, sha256(&d1));
        assert_ne!(key2, key);
    }

    #[test]
    fn evp_bytes_to_key_degenerate_outputs() {
        // Zero-length key and iv: must terminate immediately, writing nothing.
        evp_bytes_to_key(b"pw", Some(&[0u8; 8]), 1, &mut [], &mut []);

        // key only (iv empty) — first 32 digest bytes.
        let mut key = [0u8; AES_256_KEY_LEN];
        evp_bytes_to_key(b"pw", Some(&[0u8; 8]), 1, &mut key, &mut []);
        let mut both_key = [0u8; AES_256_KEY_LEN];
        let mut both_iv = [0u8; AES_256_CBC_IV_LEN];
        evp_bytes_to_key(b"pw", Some(&[0u8; 8]), 1, &mut both_key, &mut both_iv);
        assert_eq!(key, both_key);

        // iv only — the iv is then the FIRST digest bytes, since key takes none.
        let mut iv = [0u8; AES_256_CBC_IV_LEN];
        evp_bytes_to_key(b"pw", Some(&[0u8; 8]), 1, &mut [], &mut iv);
        assert_eq!(iv[..], both_key[..AES_256_CBC_IV_LEN]);

        // An output longer than one digest spills into D2, D3, ...
        let mut long = [0u8; 96];
        evp_bytes_to_key(b"pw", Some(&[0u8; 8]), 1, &mut long, &mut []);
        assert_eq!(&long[..32], &both_key[..]);
        assert_ne!(&long[32..64], &long[..32]);
    }

    // -- salt handling (divergence 4) ---------------------------------------

    #[test]
    fn salt_pkcs5_pads_truncates_and_defaults_to_zeros() {
        let mut aes = AesDefault::new();
        assert_eq!(aes.salt(), b"");
        assert_eq!(aes.salt_pkcs5(), [0u8; 8], "empty salt_ -> 8 zero bytes");

        let mut rng = CounterRng::new(0);
        aes.create_initial_vector(b"abc", &mut rng).unwrap();
        assert_eq!(aes.salt(), b"abc");
        assert_eq!(
            aes.salt_pkcs5(),
            *b"abc\0\0\0\0\0",
            "short salt zero-padded"
        );

        aes.create_initial_vector(b"01234567", &mut rng).unwrap();
        assert_eq!(aes.salt_pkcs5(), *b"01234567", "exact-length salt");

        aes.create_initial_vector(b"0123456789abcdef", &mut rng)
            .unwrap();
        assert_eq!(aes.salt_pkcs5(), *b"01234567", "long salt truncated to 8");

        // Salt is replaced, not appended, across calls.
        aes.create_initial_vector(b"", &mut rng).unwrap();
        assert_eq!(aes.salt(), b"");
        assert_eq!(aes.salt_pkcs5(), [0u8; 8]);
    }

    // -- ctp::AES key/IV state machine --------------------------------------

    #[test]
    fn generate_key_matches_openssl_for_default_empty_salt() {
        // The order test_encrypt.cc and the docs use: GenerateKey("passwd") on a
        // fresh object, i.e. salt_ == "" -> zero salt here.
        let aes = keyed_aes();
        assert!(aes.has_key() && aes.has_iv());
        assert_eq!(
            hex(aes.key()),
            "0f151149ec2da0732ef63eec89d45b413f07eb9331a060c27e9c6816a69bbe46"
        );
        // GenerateKey also writes the DERIVED iv (divergence 5).
        assert_eq!(hex(aes.iv()), "1ed2bfe61f5ffe027f95143ca4f58df8");
    }

    #[test]
    fn generate_key_then_create_initial_vector_keeps_random_iv() {
        // Documented order: the random IV overwrites the derived one, and the
        // salt never reaches the key derivation.
        let mut aes = AesDefault::new();
        aes.generate_key(b"passwd");
        let derived_iv = *aes.iv();
        let key_before = *aes.key();

        let mut rng = CounterRng::new(0xA0);
        aes.create_initial_vector(b"0102030405060708", &mut rng)
            .unwrap();

        assert_eq!(
            *aes.key(),
            key_before,
            "key untouched by CreateInitialVector"
        );
        assert_ne!(*aes.iv(), derived_iv, "random IV replaced the derived one");
        assert_eq!(
            *aes.iv(),
            [
                0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xAB, 0xAC, 0xAD,
                0xAE, 0xAF
            ]
        );
        assert_eq!(aes.iv().len(), AES_256_CBC_IV_LEN);
    }

    #[test]
    fn create_initial_vector_then_generate_key_keeps_derived_iv() {
        // Reverse order: the salt IS used, and the derived IV silently discards
        // the random one (divergence 5).
        let mut aes = AesDefault::new();
        let mut rng = CounterRng::new(0xA0);
        aes.create_initial_vector(&[1, 2, 3, 4, 5, 6, 7, 8], &mut rng)
            .unwrap();
        let random_iv = *aes.iv();
        aes.generate_key(b"passwd");

        // Salted key/iv == the openssl -S 0102030405060708 vector.
        assert_eq!(
            hex(aes.key()),
            "4af221335492ba211823e8dc89e76697f0f934836e95a155e838fc783091d9bc"
        );
        assert_eq!(hex(aes.iv()), "17600c44c1385a55b1135509d4efb143");
        assert_ne!(*aes.iv(), random_iv, "derived IV discarded the random one");
    }

    #[test]
    fn rng_failure_propagates_instead_of_zero_iv() {
        // The C++ ignores RAND_bytes's return code and keeps an all-zero IV.
        let mut aes = AesDefault::new();
        let mut rng = CounterRng::failing();
        assert_eq!(
            aes.create_initial_vector(b"", &mut rng),
            Err(CryptoError::RandomSourceFailed)
        );
        assert!(!aes.has_iv(), "IV must not be considered set after failure");
    }

    #[test]
    fn set_key_and_set_iv_validate_lengths() {
        let mut aes = AesDefault::new();
        assert!(!aes.has_key() && !aes.has_iv());

        assert_eq!(
            aes.set_key(&[0u8; 31]),
            Err(CryptoError::BadLength {
                what: "key",
                expected: 32,
                got: 31
            })
        );
        assert_eq!(
            aes.set_key(&[]),
            Err(CryptoError::BadLength {
                what: "key",
                expected: 32,
                got: 0
            })
        );
        assert!(
            !aes.has_key(),
            "failed set_key must not mark the key as set"
        );

        assert!(aes.set_key(&[7u8; 32]).is_ok());
        assert_eq!(*aes.key(), [7u8; 32]);
        assert!(aes.has_key());

        assert_eq!(
            aes.set_iv(&[0u8; 17]),
            Err(CryptoError::BadLength {
                what: "iv",
                expected: 16,
                got: 17
            })
        );
        assert!(aes.set_iv(&[3u8; 16]).is_ok());
        assert_eq!(*aes.iv(), [3u8; 16]);
    }

    // -- length math --------------------------------------------------------

    #[test]
    fn ciphertext_len_matches_openssl() {
        // Confirmed against `openssl enc -aes-256-cbc` output sizes.
        for (pt, ct) in [
            (0usize, 16usize),
            (1, 16),
            (10, 16),
            (15, 16),
            (16, 32),
            (17, 32),
            (31, 32),
            (32, 48),
            (8192, 8208),
        ] {
            assert_eq!(pkcs7_ciphertext_len(pt).unwrap(), ct, "plaintext {pt}");
            assert_eq!(Aes::<FakeBlockCipher>::encrypt_output_len(pt).unwrap(), ct);
        }
    }

    #[test]
    fn ciphertext_len_overflow_saturates_to_error() {
        // usize::MAX rounds down to a multiple of 16, then + 16 overflows.
        assert_eq!(
            pkcs7_ciphertext_len(usize::MAX),
            Err(CryptoError::LengthOverflow)
        );
        assert_eq!(
            pkcs7_ciphertext_len(usize::MAX - 15),
            Err(CryptoError::LengthOverflow)
        );
        // The largest input that still fits.
        let ok = usize::MAX - AES_BLOCK_SIZE - (usize::MAX % AES_BLOCK_SIZE) + 1;
        assert!(pkcs7_ciphertext_len(ok).is_ok());
    }

    // -- PKCS#7 -------------------------------------------------------------

    #[test]
    fn pkcs7_unpad_accepts_valid_and_rejects_malformed() {
        // Full-block padding (a block-aligned plaintext) -> keep 0 bytes.
        assert_eq!(pkcs7_unpad_len(&[16u8; 16]).unwrap(), 0);
        // One pad byte -> keep 15.
        let mut b = [0xAAu8; 16];
        b[15] = 1;
        assert_eq!(pkcs7_unpad_len(&b).unwrap(), 15);
        // Pad of 5.
        let mut b = [0xAAu8; 16];
        for slot in b[11..].iter_mut() {
            *slot = 5;
        }
        assert_eq!(pkcs7_unpad_len(&b).unwrap(), 11);

        // pad == 0 is invalid.
        assert_eq!(pkcs7_unpad_len(&[0u8; 16]), Err(CryptoError::BadPadding));
        // pad > block size is invalid.
        let mut b = [0u8; 16];
        b[15] = 17;
        assert_eq!(pkcs7_unpad_len(&b), Err(CryptoError::BadPadding));
        b[15] = 0xFF;
        assert_eq!(pkcs7_unpad_len(&b), Err(CryptoError::BadPadding));
        // Inconsistent padding bytes are invalid.
        let mut b = [16u8; 16];
        b[0] = 15;
        assert_eq!(pkcs7_unpad_len(&b), Err(CryptoError::BadPadding));
    }

    // -- CBC round trips ----------------------------------------------------

    #[test]
    fn encrypt_decrypt_round_trip_all_boundaries() {
        let aes = keyed_aes();
        for n in [0usize, 1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 255, 256, 1024] {
            let plaintext: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
            let ct_len = Aes::<FakeBlockCipher>::encrypt_output_len(n).unwrap();

            let mut ct = vec![0u8; ct_len];
            let written = aes.encrypt(&mut ct, &plaintext).unwrap();
            assert_eq!(written, ct_len, "ciphertext length for {n}");

            let mut pt = vec![0u8; written];
            let out = aes.decrypt(&mut pt, &ct[..written]).unwrap();
            assert_eq!(out, n, "plaintext length for {n}");
            assert_eq!(&pt[..out], &plaintext[..], "round trip for {n}");
        }
    }

    #[test]
    fn cpp_unit_test_replica() {
        // test_encrypt.cc: 8192 zero bytes, oversized buffers, one AES object.
        // This is the only size the C++ test covers — and the only size for
        // which the C++ Encrypt/Decrypt length handling is correct.
        let aes = keyed_aes();
        let data = vec![0u8; 8192];
        let mut encoded = vec![1u8; 8192 + 256];
        let mut decoded = vec![2u8; 8192 + 256];

        let encoded_size = aes.encrypt(&mut encoded, &data).unwrap();
        assert_eq!(encoded_size, 8208, "matches OpenSSL's 8192 -> 8208");
        let decoded_size = aes.decrypt(&mut decoded, &encoded[..encoded_size]).unwrap();
        assert_eq!(decoded_size, 8192);
        assert_eq!(&decoded[..decoded_size], &data[..]);
    }

    #[test]
    fn cbc_chains_blocks_and_depends_on_iv() {
        // Identical plaintext blocks must NOT produce identical ciphertext
        // blocks — that is the whole point of CBC over ECB.
        let aes = keyed_aes();
        let plaintext = vec![0x42u8; 64];
        let mut ct = vec![0u8; 80];
        aes.encrypt(&mut ct, &plaintext).unwrap();
        assert_ne!(&ct[0..16], &ct[16..32], "CBC must chain (not ECB)");

        // Changing only the IV changes the whole ciphertext.
        let mut aes2 = keyed_aes();
        let mut iv = *aes.iv();
        iv[0] ^= 0x01;
        aes2.set_iv(&iv).unwrap();
        let mut ct2 = vec![0u8; 80];
        aes2.encrypt(&mut ct2, &plaintext).unwrap();
        assert_ne!(ct, ct2, "IV must affect the ciphertext");

        // And a wrong IV corrupts (at least) the first block on decrypt.
        let mut pt = vec![0u8; ct.len()];
        let n = aes2.decrypt(&mut pt, &ct).unwrap();
        assert_ne!(&pt[..16], &plaintext[..16], "wrong IV corrupts block 0");
        // CBC self-synchronises: later blocks still recover.
        assert_eq!(&pt[16..n], &plaintext[16..], "later blocks unaffected");
    }

    #[test]
    fn encrypt_rejects_undersized_output() {
        // The C++ writes past the end of the buffer here instead.
        let aes = keyed_aes();
        let plaintext = [0u8; 16];
        let mut too_small = [0u8; 31]; // needs 32
        assert_eq!(
            aes.encrypt(&mut too_small, &plaintext),
            Err(CryptoError::OutputTooSmall {
                needed: 32,
                got: 31
            })
        );
        // Empty input still needs a full padding block.
        let mut none = [0u8; 0];
        assert_eq!(
            aes.encrypt(&mut none, &[]),
            Err(CryptoError::OutputTooSmall { needed: 16, got: 0 })
        );
        // Exactly-sized is fine; oversized is fine and leaves the tail alone.
        let mut exact = [0u8; 32];
        assert_eq!(aes.encrypt(&mut exact, &plaintext).unwrap(), 32);
        let mut over = [0xEEu8; 48];
        assert_eq!(aes.encrypt(&mut over, &plaintext).unwrap(), 32);
        assert_eq!(
            &over[32..],
            &[0xEEu8; 16],
            "tail beyond the ciphertext kept"
        );
    }

    #[test]
    fn decrypt_rejects_bad_lengths_and_buffers() {
        let aes = keyed_aes();
        let mut out = [0u8; 64];

        // Empty ciphertext: EVP_DecryptFinal_ex fails -> C++ returns false.
        assert_eq!(
            aes.decrypt(&mut out, &[]),
            Err(CryptoError::InvalidCiphertextLen(0))
        );
        // Not block aligned.
        assert_eq!(
            aes.decrypt(&mut out, &[0u8; 17]),
            Err(CryptoError::InvalidCiphertextLen(17))
        );
        assert_eq!(
            aes.decrypt(&mut out, &[0u8; 15]),
            Err(CryptoError::InvalidCiphertextLen(15))
        );
        // Output smaller than the ciphertext (the documented C++ contract).
        let mut ct = [0u8; 32];
        aes.encrypt(&mut ct, &[9u8; 16]).unwrap();
        let mut small = [0u8; 31];
        assert_eq!(
            aes.decrypt(&mut small, &ct),
            Err(CryptoError::OutputTooSmall {
                needed: 32,
                got: 31
            })
        );
    }

    #[test]
    fn decrypt_detects_tampering_and_wrong_key() {
        let aes = keyed_aes();
        let plaintext = [7u8; 40];
        let mut ct = vec![0u8; 48];
        let n = aes.encrypt(&mut ct, &plaintext).unwrap();

        // Corrupting the whole final ciphertext block destroys the PKCS#7
        // trailer, which cbc_decrypt must reject.
        //
        // NOTE: this deliberately corrupts the entire block rather than a single
        // byte. Detecting a *single-byte* flip anywhere in the block relies on
        // the cipher's avalanche/diffusion, which FakeBlockCipher (a byte
        // permutation + XOR) does not have by construction — under it, flipping
        // the last ciphertext byte perturbs exactly one plaintext byte and can
        // leave the padding bytes intact. That property belongs to real AES and
        // is the backend's to provide (divergence 1); what this port owns, and
        // what is tested here, is the padding validation itself.
        let mut tampered = ct.clone();
        for b in tampered[n - AES_BLOCK_SIZE..n].iter_mut() {
            *b ^= 0xFF;
        }
        let mut out = vec![0u8; n];
        assert_eq!(
            aes.decrypt(&mut out, &tampered[..n]),
            Err(CryptoError::BadPadding)
        );

        // A different password derives a different key -> padding check fails.
        let mut other = Aes::<FakeBlockCipher>::new();
        other.generate_key(b"wrong-password");
        other.set_iv(aes.iv()).unwrap();
        assert_eq!(
            other.decrypt(&mut out, &ct[..n]),
            Err(CryptoError::BadPadding)
        );

        // Same password + same IV round-trips across objects (sender/receiver).
        let mut receiver = Aes::<FakeBlockCipher>::new();
        receiver.generate_key(b"passwd");
        receiver.set_iv(aes.iv()).unwrap();
        let got = receiver.decrypt(&mut out, &ct[..n]).unwrap();
        assert_eq!(&out[..got], &plaintext[..]);
    }

    #[test]
    fn encrypt_requires_key_and_iv() {
        // C++ hands empty std::strings to EVP here (UB); this port errors.
        let fresh = Aes::<FakeBlockCipher>::new();
        let mut out = [0u8; 32];
        assert_eq!(
            fresh.encrypt(&mut out, &[1u8; 4]),
            Err(CryptoError::KeyNotSet)
        );
        assert_eq!(
            fresh.decrypt(&mut out, &[0u8; 16]),
            Err(CryptoError::KeyNotSet)
        );

        let mut keyed = Aes::<FakeBlockCipher>::new();
        keyed.set_key(&[1u8; 32]).unwrap();
        assert_eq!(
            keyed.encrypt(&mut out, &[1u8; 4]),
            Err(CryptoError::IvNotSet)
        );

        keyed.set_iv(&[0u8; 16]).unwrap();
        assert!(keyed.encrypt(&mut out, &[1u8; 4]).is_ok());
    }

    // -- backend seam -------------------------------------------------------

    #[test]
    fn default_backend_reports_cipher_unavailable() {
        // Key derivation works with no cipher linked; encryption does not.
        let mut aes = AesDefault::new();
        aes.generate_key(b"passwd");
        assert!(aes.has_key());

        let mut out = [0u8; 32];
        assert_eq!(
            aes.encrypt(&mut out, &[0u8; 4]),
            Err(CryptoError::CipherUnavailable)
        );
        assert_eq!(
            aes.decrypt(&mut out, &[0u8; 16]),
            Err(CryptoError::CipherUnavailable)
        );
        assert!(UnavailableAes256::new(&[0u8; 32]).is_err());
    }

    #[test]
    fn cipher_unavailable_is_checked_after_key_state() {
        // Missing key/IV is reported before the missing backend, so the state
        // errors stay diagnosable without a cipher.
        let fresh = AesDefault::new();
        let mut out = [0u8; 32];
        assert_eq!(fresh.encrypt(&mut out, &[]), Err(CryptoError::KeyNotSet));
    }

    // -- misc ---------------------------------------------------------------

    #[test]
    fn errors_display_without_panicking() {
        let errs = [
            CryptoError::CipherUnavailable,
            CryptoError::KeyNotSet,
            CryptoError::IvNotSet,
            CryptoError::BadLength {
                what: "key",
                expected: 32,
                got: 3,
            },
            CryptoError::OutputTooSmall { needed: 16, got: 1 },
            CryptoError::InvalidCiphertextLen(17),
            CryptoError::BadPadding,
            CryptoError::LengthOverflow,
            CryptoError::RandomSourceFailed,
        ];
        for e in &errs {
            assert!(!e.to_string().is_empty());
        }
    }

    #[test]
    fn aes_is_send_and_sync_and_usable_concurrently() {
        // Project rule: no thread_local. Aes carries all state explicitly, so a
        // keyed instance shares across threads behind an Arc with no interior
        // mutability; encrypt/decrypt take &self.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Aes<FakeBlockCipher>>();
        assert_send_sync::<AesDefault>();
        assert_send_sync::<CryptoError>();

        use std::sync::Arc;
        let aes = Arc::new(keyed_aes());
        let handles: Vec<_> = (0..8u8)
            .map(|t| {
                let aes = Arc::clone(&aes);
                std::thread::spawn(move || {
                    let plaintext = vec![t; 100 + t as usize];
                    let mut ct = vec![0u8; 128];
                    let n = aes.encrypt(&mut ct, &plaintext).unwrap();
                    let mut pt = vec![0u8; n];
                    let m = aes.decrypt(&mut pt, &ct[..n]).unwrap();
                    assert_eq!(&pt[..m], &plaintext[..]);
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn hex_helpers_round_trip() {
        // Guards the vector comparisons above against a broken test helper.
        assert_eq!(hex(&unhex("00ff10")), "00ff10");
        assert_eq!(unhex("deadbeef"), vec![0xde, 0xad, 0xbe, 0xef]);
        assert_eq!(hex(&[]), "");
    }
}
