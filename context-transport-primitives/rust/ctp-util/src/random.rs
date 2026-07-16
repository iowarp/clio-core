// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).

//! Random-number distribution wrappers ported from
//! `include/clio_ctp/util/random.h`.
//!
//! # C++ -> Rust name mapping
//!
//! | C++ (`ctp::`)                              | Rust                                      |
//! |--------------------------------------------|-------------------------------------------|
//! | `Distribution` (abstract base)             | [`Distribution`] (trait)                  |
//! | `Distribution::Seed()`                     | [`Distribution::seed_from_time`]          |
//! | `Distribution::Seed(size_t)`               | [`Distribution::seed`]                    |
//! | `Distribution::GetInt()`                   | [`Distribution::get_int`]                 |
//! | `Distribution::GetDouble()`                | [`Distribution::get_double`]              |
//! | `Distribution::GetSize()`                  | [`Distribution::get_size`]                |
//! | `CountDistribution`                        | [`CountDistribution`]                     |
//! | `CountDistribution::Shape(size_t)`         | [`CountDistribution::shape`]              |
//! | `NormalDistribution`                       | [`NormalDistribution`]                    |
//! | `NormalDistribution::Shape(double)`        | [`NormalDistribution::shape_std`]         |
//! | `NormalDistribution::Shape(double,double)` | [`NormalDistribution::shape`]             |
//! | `GammaDistribution`                        | [`GammaDistribution`]                     |
//! | `GammaDistribution::Shape(double)`         | [`GammaDistribution::shape_scale`]        |
//! | `GammaDistribution::Shape(double,double)`  | [`GammaDistribution::shape`]              |
//! | `ExponentialDistribution`                  | [`ExponentialDistribution`]               |
//! | `ExponentialDistribution::Shape(double)`   | [`ExponentialDistribution::shape`]        |
//! | `UniformDistribution`                      | [`UniformDistribution`]                   |
//! | `UniformDistribution::Shape(high)`         | [`UniformDistribution::shape_high`]       |
//! | `UniformDistribution::Shape(low,high)`     | [`UniformDistribution::shape`]            |
//!
//! # Semantic divergences (IMPORTANT — reproducibility)
//!
//! * **Different PRNG stream.** The C++ code uses `std::default_random_engine`,
//!   which is *implementation-defined* (`minstd_rand0` on libstdc++,
//!   `mt19937` variants elsewhere), so the C++ API itself never guaranteed a
//!   reproducible cross-platform stream. This port uses **SplitMix64**
//!   (Steele/Lea/Vigna, public-domain reference algorithm), a small
//!   self-contained 64-bit PRNG. Sequences are reproducible for a given seed
//!   *within this Rust port*, but never match any C++ build.
//! * **Different sampling algorithms.** Normal uses Box–Muller (with a cached
//!   spare, cleared on re-seed); gamma uses Marsaglia–Tsang (with the
//!   `shape < 1` boost); exponential uses inversion. `std::*_distribution`
//!   algorithms are also implementation-defined in C++, so, as above, only
//!   distributional (not bitwise) parity ever existed.
//! * **Saturating float→int casts.** C++ `(int)round(d)` / `(size_t)round(d)`
//!   are undefined behavior when out of range (and negative→`size_t` wraps).
//!   Rust `as` saturates: `get_size` on a negative sample returns 0, and
//!   out-of-range values clamp to the integer type's min/max.
//! * **`Seed()` (no-arg) time source.** C++ seeds from
//!   `std::chrono::steady_clock`; the port seeds from `SystemTime` nanoseconds
//!   since the Unix epoch (folded to 64 bits). Both are "nondeterministic
//!   time-based seed"; the exact value differs.
//! * **No overloading in Rust.** The `Seed`/`Shape` overload sets become
//!   distinctly named methods, per the table above.
//! * **`CountDistribution` counter width.** The counter is `usize`, exactly as
//!   the C++ `size_t`; `get_int` truncates it to `i32` (two's-complement
//!   wrap), matching the C++ `size_t -> int` conversion on all mainstream
//!   implementations.
//! * `UniformDistribution::Shape(size_t high)` and `Shape(double high)` are
//!   one C++ overload pair that both produce `[0, high)`; the port exposes a
//!   single [`UniformDistribution::shape_high`] taking `f64` (call it with
//!   `n as f64` for the `size_t` flavor).

use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// PRNG core: SplitMix64
// ---------------------------------------------------------------------------

/// SplitMix64 pseudo-random engine (reference constants from Vigna's
/// public-domain implementation). Small, fast, passes BigCrush; used here in
/// place of the implementation-defined `std::default_random_engine`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f64` in `[0, 1)` with 53 bits of precision.
    fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / (1u64 << 53) as f64;
        (self.next_u64() >> 11) as f64 * SCALE
    }

    /// Uniform `f64` in `(0, 1]` — never zero, safe for `ln()`.
    fn next_f64_open_zero(&mut self) -> f64 {
        1.0 - self.next_f64()
    }
}

/// Time-based seed value (C++ `steady_clock::now().time_since_epoch().count()`
/// analogue; see the divergence notes in the module docs).
fn time_seed() -> u64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => {
            let n = d.as_nanos();
            (n as u64) ^ ((n >> 64) as u64)
        }
        // Clock before the epoch: still produce *some* nonzero entropy.
        Err(e) => e.duration().as_nanos() as u64 | 1,
    }
}

/// Rounds to nearest (ties away from zero, like C `round`) then converts,
/// saturating at the target type's bounds (divergence: C++ casts are UB when
/// out of range).
fn round_to_i32(x: f64) -> i32 {
    x.round() as i32
}

/// See [`round_to_i32`]; negative inputs saturate to 0.
fn round_to_usize(x: f64) -> usize {
    x.round() as usize
}

/// One standard-normal (mean 0, std 1) variate via Box–Muller, returning the
/// primary value and the spare to cache.
fn box_muller(rng: &mut SplitMix64) -> (f64, f64) {
    let u1 = rng.next_f64_open_zero(); // in (0, 1]; ln() is finite
    let u2 = rng.next_f64();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = std::f64::consts::TAU * u2;
    (r * theta.cos(), r * theta.sin())
}

// ---------------------------------------------------------------------------
// Distribution trait (C++ `ctp::Distribution` abstract base)
// ---------------------------------------------------------------------------

/// Port of the C++ `ctp::Distribution` abstract base class.
///
/// All sampling methods take `&mut self` because drawing advances the
/// underlying engine state (as in C++, where `operator()` mutates the
/// engine).
pub trait Distribution {
    /// C++ `Seed()`: re-seed the engine from the current time
    /// (nondeterministic).
    fn seed_from_time(&mut self);
    /// C++ `Seed(size_t seed)`: re-seed the engine deterministically.
    fn seed(&mut self, seed: u64);
    /// C++ `GetInt()`: next sample rounded/converted to `i32`.
    fn get_int(&mut self) -> i32;
    /// C++ `GetDouble()`: next sample as `f64`.
    fn get_double(&mut self) -> f64;
    /// C++ `GetSize()`: next sample rounded/converted to `usize`.
    fn get_size(&mut self) -> usize;
}

// ---------------------------------------------------------------------------
// CountDistribution
// ---------------------------------------------------------------------------

/// Port of C++ `ctp::CountDistribution`: a deterministic counter that starts
/// at 0 and advances by a configurable increment (default 1) on every call.
#[derive(Debug, Clone, Default)]
pub struct CountDistribution {
    inc: usize,
    count: usize,
    initialized: bool,
}

impl CountDistribution {
    /// Equivalent to C++ default construction (`inc_ = 1`, `count_ = 0`).
    pub fn new() -> Self {
        Self {
            inc: 1,
            count: 0,
            initialized: true,
        }
    }

    /// C++ `Shape(size_t inc)`: set the increment.
    pub fn shape(&mut self, inc: usize) {
        self.ensure_init();
        self.inc = inc;
    }

    /// `Default` cannot express `inc = 1`, so lazily fix it up. Any instance
    /// built via `CountDistribution::new()` is already initialized.
    fn ensure_init(&mut self) {
        if !self.initialized {
            self.inc = 1;
            self.initialized = true;
        }
    }

    fn advance(&mut self) -> usize {
        self.ensure_init();
        let temp = self.count;
        self.count = self.count.wrapping_add(self.inc);
        temp
    }
}

impl Distribution for CountDistribution {
    /// No-op: matches C++, where re-seeding the engine does not affect the
    /// counter (the engine is unused by this distribution).
    fn seed_from_time(&mut self) {}

    /// No-op: see [`CountDistribution::seed_from_time`].
    fn seed(&mut self, _seed: u64) {}

    fn get_int(&mut self) -> i32 {
        // C++: `int temp = count_;` — size_t truncated to int (wrapping).
        self.advance() as i32
    }

    fn get_double(&mut self) -> f64 {
        self.advance() as f64
    }

    fn get_size(&mut self) -> usize {
        self.advance()
    }
}

// ---------------------------------------------------------------------------
// NormalDistribution
// ---------------------------------------------------------------------------

/// Port of C++ `ctp::NormalDistribution` (`std::normal_distribution<double>`).
/// Defaults to mean 0, standard deviation 1.
#[derive(Debug, Clone)]
pub struct NormalDistribution {
    rng: SplitMix64,
    mean: f64,
    stddev: f64,
    /// Cached second Box–Muller variate (standard-normal units).
    spare: Option<f64>,
}

impl Default for NormalDistribution {
    fn default() -> Self {
        Self::new()
    }
}

impl NormalDistribution {
    pub fn new() -> Self {
        Self {
            rng: SplitMix64::new(0),
            mean: 0.0,
            stddev: 1.0,
            spare: None,
        }
    }

    /// C++ `Shape(double std)`: mean 0, standard deviation `std`.
    pub fn shape_std(&mut self, std: f64) {
        self.shape(0.0, std);
    }

    /// C++ `Shape(double mean, double std)`.
    pub fn shape(&mut self, mean: f64, std: f64) {
        self.mean = mean;
        self.stddev = std;
        self.spare = None;
    }

    fn sample(&mut self) -> f64 {
        let z = match self.spare.take() {
            Some(z) => z,
            None => {
                let (z0, z1) = box_muller(&mut self.rng);
                self.spare = Some(z1);
                z0
            }
        };
        self.mean + self.stddev * z
    }
}

impl Distribution for NormalDistribution {
    fn seed_from_time(&mut self) {
        self.seed(time_seed());
    }

    fn seed(&mut self, seed: u64) {
        self.rng = SplitMix64::new(seed);
        self.spare = None;
    }

    fn get_int(&mut self) -> i32 {
        round_to_i32(self.sample())
    }

    fn get_double(&mut self) -> f64 {
        self.sample()
    }

    fn get_size(&mut self) -> usize {
        round_to_usize(self.sample())
    }
}

// ---------------------------------------------------------------------------
// GammaDistribution
// ---------------------------------------------------------------------------

/// Port of C++ `ctp::GammaDistribution` (`std::gamma_distribution<double>`),
/// parameterized by shape `k` and scale `theta` (mean = `k * theta`).
/// Defaults to shape 1, scale 1. Sampled with the Marsaglia–Tsang method.
#[derive(Debug, Clone)]
pub struct GammaDistribution {
    rng: SplitMix64,
    shape: f64,
    scale: f64,
}

impl Default for GammaDistribution {
    fn default() -> Self {
        Self::new()
    }
}

impl GammaDistribution {
    pub fn new() -> Self {
        Self {
            rng: SplitMix64::new(0),
            shape: 1.0,
            scale: 1.0,
        }
    }

    /// C++ `Shape(double scale)`: shape 1, scale `scale`.
    pub fn shape_scale(&mut self, scale: f64) {
        self.shape(1.0, scale);
    }

    /// C++ `Shape(double shape, double scale)`.
    pub fn shape(&mut self, shape: f64, scale: f64) {
        self.shape = shape;
        self.scale = scale;
    }

    fn sample(&mut self) -> f64 {
        let a = self.shape;
        if a < 1.0 {
            // Boost: Gamma(a) = Gamma(a + 1) * U^(1/a).
            let u = self.rng.next_f64_open_zero();
            return Self::sample_shape_ge1(&mut self.rng, a + 1.0) * u.powf(1.0 / a) * self.scale;
        }
        Self::sample_shape_ge1(&mut self.rng, a) * self.scale
    }

    /// Marsaglia–Tsang (2000) for shape `a >= 1`; returns a Gamma(a, 1)
    /// variate.
    fn sample_shape_ge1(rng: &mut SplitMix64, a: f64) -> f64 {
        let d = a - 1.0 / 3.0;
        let c = 1.0 / (3.0 * d.sqrt());
        loop {
            let (x, _) = box_muller(rng);
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v = v * v * v;
            let u = rng.next_f64_open_zero();
            if u < 1.0 - 0.0331 * (x * x) * (x * x) {
                return d * v;
            }
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }
}

impl Distribution for GammaDistribution {
    fn seed_from_time(&mut self) {
        self.seed(time_seed());
    }

    fn seed(&mut self, seed: u64) {
        self.rng = SplitMix64::new(seed);
    }

    fn get_int(&mut self) -> i32 {
        round_to_i32(self.sample())
    }

    fn get_double(&mut self) -> f64 {
        self.sample()
    }

    fn get_size(&mut self) -> usize {
        round_to_usize(self.sample())
    }
}

// ---------------------------------------------------------------------------
// ExponentialDistribution
// ---------------------------------------------------------------------------

/// Port of C++ `ctp::ExponentialDistribution`
/// (`std::exponential_distribution<double>`). The `Shape` parameter is the
/// **rate** lambda (the C++ code names it `scale`, but it is passed straight
/// to `std::exponential_distribution`, whose parameter is the rate; mean =
/// `1 / lambda`). Defaults to lambda = 1.
#[derive(Debug, Clone)]
pub struct ExponentialDistribution {
    rng: SplitMix64,
    lambda: f64,
}

impl Default for ExponentialDistribution {
    fn default() -> Self {
        Self::new()
    }
}

impl ExponentialDistribution {
    pub fn new() -> Self {
        Self {
            rng: SplitMix64::new(0),
            lambda: 1.0,
        }
    }

    /// C++ `Shape(double scale)`: set the rate lambda (see type docs).
    pub fn shape(&mut self, scale: f64) {
        self.lambda = scale;
    }

    fn sample(&mut self) -> f64 {
        // Inversion: -ln(U) / lambda with U in (0, 1].
        -self.rng.next_f64_open_zero().ln() / self.lambda
    }
}

impl Distribution for ExponentialDistribution {
    fn seed_from_time(&mut self) {
        self.seed(time_seed());
    }

    fn seed(&mut self, seed: u64) {
        self.rng = SplitMix64::new(seed);
    }

    fn get_int(&mut self) -> i32 {
        round_to_i32(self.sample())
    }

    fn get_double(&mut self) -> f64 {
        self.sample()
    }

    fn get_size(&mut self) -> usize {
        round_to_usize(self.sample())
    }
}

// ---------------------------------------------------------------------------
// UniformDistribution
// ---------------------------------------------------------------------------

/// Port of C++ `ctp::UniformDistribution`
/// (`std::uniform_real_distribution<double>`): uniform reals in
/// `[low, high)`. Defaults to `[0, 1)`.
#[derive(Debug, Clone)]
pub struct UniformDistribution {
    rng: SplitMix64,
    low: f64,
    high: f64,
}

impl Default for UniformDistribution {
    fn default() -> Self {
        Self::new()
    }
}

impl UniformDistribution {
    pub fn new() -> Self {
        Self {
            rng: SplitMix64::new(0),
            low: 0.0,
            high: 1.0,
        }
    }

    /// C++ `Shape(size_t high)` / `Shape(double high)`: uniform on
    /// `[0, high)`. (For the `size_t` overload, pass `n as f64`.)
    pub fn shape_high(&mut self, high: f64) {
        self.shape(0.0, high);
    }

    /// C++ `Shape(double low, double high)`: uniform on `[low, high)`.
    pub fn shape(&mut self, low: f64, high: f64) {
        self.low = low;
        self.high = high;
    }

    fn sample(&mut self) -> f64 {
        self.low + (self.high - self.low) * self.rng.next_f64()
    }
}

impl Distribution for UniformDistribution {
    fn seed_from_time(&mut self) {
        self.seed(time_seed());
    }

    fn seed(&mut self, seed: u64) {
        self.rng = SplitMix64::new(seed);
    }

    fn get_int(&mut self) -> i32 {
        round_to_i32(self.sample())
    }

    fn get_double(&mut self) -> f64 {
        self.sample()
    }

    fn get_size(&mut self) -> usize {
        round_to_usize(self.sample())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- PRNG core ----

    #[test]
    fn splitmix64_known_answer_vectors() {
        // Reference outputs of Vigna's splitmix64 for seed = 0.
        let mut rng = SplitMix64::new(0);
        assert_eq!(rng.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(rng.next_u64(), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(rng.next_u64(), 0x06C4_5D18_8009_454F);
        // And for seed = 1234567.
        let mut rng = SplitMix64::new(1234567);
        assert_eq!(rng.next_u64(), 0x599E_D017_FB08_FC85);
    }

    #[test]
    fn splitmix64_f64_in_unit_interval() {
        let mut rng = SplitMix64::new(42);
        for _ in 0..10_000 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x), "next_f64 out of [0,1): {x}");
            let y = rng.next_f64_open_zero();
            assert!(y > 0.0 && y <= 1.0, "next_f64_open_zero out of (0,1]: {y}");
        }
    }

    // ---- rounding/conversion helpers ----

    #[test]
    fn rounding_matches_c_round_semantics() {
        assert_eq!(round_to_i32(2.5), 3); // ties away from zero
        assert_eq!(round_to_i32(-2.5), -3);
        assert_eq!(round_to_i32(2.4), 2);
        assert_eq!(round_to_i32(-0.4), 0);
        assert_eq!(round_to_usize(0.5), 1);
        assert_eq!(round_to_usize(0.49), 0);
    }

    #[test]
    fn rounding_saturates_instead_of_ub() {
        // Documented divergence: saturation, not UB/wrap.
        assert_eq!(round_to_usize(-123.7), 0);
        assert_eq!(round_to_i32(1e300), i32::MAX);
        assert_eq!(round_to_i32(-1e300), i32::MIN);
        assert_eq!(round_to_usize(1e300), usize::MAX);
        assert_eq!(round_to_i32(f64::NAN), 0);
        assert_eq!(round_to_usize(f64::NAN), 0);
    }

    // ---- CountDistribution ----

    #[test]
    fn count_default_increments_by_one() {
        let mut d = CountDistribution::new();
        assert_eq!(d.get_int(), 0);
        assert_eq!(d.get_int(), 1);
        assert_eq!(d.get_size(), 2);
        assert_eq!(d.get_double(), 3.0);
        assert_eq!(d.get_int(), 4);
    }

    #[test]
    fn count_default_trait_impl_matches_new() {
        // `Default` must behave like `new()` (inc = 1), despite derive
        // producing zeroed fields.
        let mut d = CountDistribution::default();
        assert_eq!(d.get_size(), 0);
        assert_eq!(d.get_size(), 1);
        assert_eq!(d.get_size(), 2);
    }

    #[test]
    fn count_shape_sets_increment() {
        let mut d = CountDistribution::new();
        d.shape(5);
        assert_eq!(d.get_size(), 0);
        assert_eq!(d.get_size(), 5);
        assert_eq!(d.get_size(), 10);
        // Shape(0): counter never advances (mirrors C++).
        d.shape(0);
        assert_eq!(d.get_size(), 15);
        assert_eq!(d.get_size(), 15);
    }

    #[test]
    fn count_seed_is_noop() {
        let mut d = CountDistribution::new();
        assert_eq!(d.get_size(), 0);
        d.seed(999);
        d.seed_from_time();
        assert_eq!(d.get_size(), 1); // counter unaffected, as in C++
    }

    #[test]
    fn count_wraps_like_cpp_size_t() {
        let mut d = CountDistribution::new();
        d.shape(usize::MAX); // count: 0, MAX, MAX-1 (wrapping), ...
        assert_eq!(d.get_size(), 0);
        assert_eq!(d.get_size(), usize::MAX);
        // usize::MAX as i32 == -1 (truncating conversion, like C++).
        assert_eq!(d.get_int(), -2);
    }

    // ---- determinism across all seeded distributions ----

    #[test]
    fn same_seed_same_sequence() {
        macro_rules! check {
            ($ty:ty) => {{
                let mut a = <$ty>::new();
                let mut b = <$ty>::new();
                a.seed(0xDEADBEEF);
                b.seed(0xDEADBEEF);
                for _ in 0..100 {
                    assert_eq!(a.get_double(), b.get_double());
                }
                // Re-seeding restarts the stream.
                let first = {
                    a.seed(7);
                    a.get_double()
                };
                a.seed(7);
                assert_eq!(a.get_double(), first);
            }};
        }
        check!(NormalDistribution);
        check!(GammaDistribution);
        check!(ExponentialDistribution);
        check!(UniformDistribution);
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = UniformDistribution::new();
        let mut b = UniformDistribution::new();
        a.seed(1);
        b.seed(2);
        let sa: Vec<f64> = (0..8).map(|_| a.get_double()).collect();
        let sb: Vec<f64> = (0..8).map(|_| b.get_double()).collect();
        assert_ne!(sa, sb);
    }

    #[test]
    fn seed_from_time_produces_samples() {
        let mut d = UniformDistribution::new();
        d.seed_from_time();
        let x = d.get_double();
        assert!((0.0..1.0).contains(&x));
    }

    // ---- NormalDistribution ----

    #[test]
    fn normal_moments_and_shape() {
        let mut d = NormalDistribution::new();
        d.seed(1);
        d.shape(10.0, 2.0);
        let n = 100_000;
        let samples: Vec<f64> = (0..n).map(|_| d.get_double()).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
        assert!((mean - 10.0).abs() < 0.05, "mean = {mean}");
        assert!((var.sqrt() - 2.0).abs() < 0.05, "std = {}", var.sqrt());
    }

    #[test]
    fn normal_shape_std_is_zero_mean() {
        let mut d = NormalDistribution::new();
        d.seed(2);
        d.shape_std(1.0);
        let n = 50_000;
        let mean = (0..n).map(|_| d.get_double()).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.05, "mean = {mean}");
    }

    #[test]
    fn normal_zero_std_is_constant() {
        let mut d = NormalDistribution::new();
        d.seed(3);
        d.shape(4.0, 0.0);
        for _ in 0..10 {
            assert_eq!(d.get_double(), 4.0);
            assert_eq!(d.get_int(), 4);
            assert_eq!(d.get_size(), 4);
        }
    }

    #[test]
    fn normal_negative_samples_saturate_get_size_to_zero() {
        let mut d = NormalDistribution::new();
        d.seed(4);
        d.shape(-1000.0, 1.0); // essentially always negative
        for _ in 0..100 {
            assert_eq!(d.get_size(), 0);
            assert!(d.get_int() < 0);
        }
    }

    #[test]
    fn normal_reseed_clears_cached_spare() {
        let mut a = NormalDistribution::new();
        let mut b = NormalDistribution::new();
        a.seed(5);
        b.seed(5);
        let _ = a.get_double(); // caches a spare in `a`
        a.seed(5); // must discard the spare
        assert_eq!(a.get_double(), b.get_double());
    }

    // ---- GammaDistribution ----

    #[test]
    fn gamma_mean_matches_shape_times_scale() {
        let mut d = GammaDistribution::new();
        d.seed(6);
        d.shape(3.0, 2.0); // mean = 6
        let n = 100_000;
        let mut sum = 0.0;
        for _ in 0..n {
            let x = d.get_double();
            assert!(x > 0.0, "gamma sample must be positive: {x}");
            sum += x;
        }
        let mean = sum / n as f64;
        assert!((mean - 6.0).abs() < 0.1, "mean = {mean}");
    }

    #[test]
    fn gamma_shape_less_than_one_branch() {
        let mut d = GammaDistribution::new();
        d.seed(7);
        d.shape(0.5, 1.0); // mean = 0.5
        let n = 100_000;
        let mut sum = 0.0;
        for _ in 0..n {
            let x = d.get_double();
            assert!(x > 0.0 && x.is_finite(), "bad gamma(0.5) sample: {x}");
            sum += x;
        }
        let mean = sum / n as f64;
        assert!((mean - 0.5).abs() < 0.05, "mean = {mean}");
    }

    #[test]
    fn gamma_shape_scale_defaults_shape_to_one() {
        // Shape(scale) == exponential with mean `scale`.
        let mut d = GammaDistribution::new();
        d.seed(8);
        d.shape_scale(4.0);
        let n = 100_000;
        let mean = (0..n).map(|_| d.get_double()).sum::<f64>() / n as f64;
        assert!((mean - 4.0).abs() < 0.1, "mean = {mean}");
    }

    // ---- ExponentialDistribution ----

    #[test]
    fn exponential_mean_is_reciprocal_rate() {
        let mut d = ExponentialDistribution::new();
        d.seed(9);
        d.shape(0.25); // mean = 4
        let n = 100_000;
        let mut sum = 0.0;
        for _ in 0..n {
            let x = d.get_double();
            assert!(x >= 0.0 && x.is_finite(), "bad exp sample: {x}");
            sum += x;
        }
        let mean = sum / n as f64;
        assert!((mean - 4.0).abs() < 0.1, "mean = {mean}");
    }

    #[test]
    fn exponential_int_and_size_round() {
        let mut d = ExponentialDistribution::new();
        d.seed(10);
        d.shape(1.0);
        for _ in 0..1000 {
            let i = d.get_int();
            assert!(i >= 0);
            let s = d.get_size();
            assert!(s < 100, "exp(1) sample rounded to {s}, implausibly large");
        }
    }

    // ---- UniformDistribution ----

    #[test]
    fn uniform_default_is_unit_interval() {
        let mut d = UniformDistribution::new();
        d.seed(11);
        for _ in 0..10_000 {
            let x = d.get_double();
            assert!((0.0..1.0).contains(&x), "out of [0,1): {x}");
        }
    }

    #[test]
    fn uniform_shape_bounds_respected() {
        let mut d = UniformDistribution::new();
        d.seed(12);
        d.shape(-5.0, 5.0);
        let n = 50_000;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut sum = 0.0;
        for _ in 0..n {
            let x = d.get_double();
            assert!((-5.0..5.0).contains(&x), "out of [-5,5): {x}");
            min = min.min(x);
            max = max.max(x);
            sum += x;
        }
        assert!((sum / n as f64).abs() < 0.05);
        assert!(min < -4.9 && max > 4.9, "range not covered: [{min}, {max}]");
    }

    #[test]
    fn uniform_shape_high_starts_at_zero() {
        let mut d = UniformDistribution::new();
        d.seed(13);
        d.shape_high(100.0);
        for _ in 0..10_000 {
            let x = d.get_double();
            assert!((0.0..100.0).contains(&x), "out of [0,100): {x}");
            let s = d.get_size();
            assert!(s <= 100, "get_size out of range: {s}");
        }
    }

    #[test]
    fn uniform_degenerate_range_is_constant() {
        let mut d = UniformDistribution::new();
        d.seed(14);
        d.shape(3.0, 3.0);
        for _ in 0..10 {
            assert_eq!(d.get_double(), 3.0);
            assert_eq!(d.get_int(), 3);
            assert_eq!(d.get_size(), 3);
        }
    }

    #[test]
    fn uniform_get_int_rounds_nearest() {
        let mut d = UniformDistribution::new();
        d.seed(15);
        d.shape(0.0, 1.0);
        // Samples in [0,1) round to 0 or 1 only.
        for _ in 0..1000 {
            let i = d.get_int();
            assert!(i == 0 || i == 1, "unexpected rounding: {i}");
        }
    }

    // ---- trait-object usability (mirrors C++ virtual dispatch) ----

    #[test]
    fn distributions_usable_as_trait_objects() {
        let mut dists: Vec<Box<dyn Distribution>> = vec![
            Box::new(CountDistribution::new()),
            Box::new(NormalDistribution::new()),
            Box::new(GammaDistribution::new()),
            Box::new(ExponentialDistribution::new()),
            Box::new(UniformDistribution::new()),
        ];
        for d in &mut dists {
            d.seed(1);
            let _ = d.get_int();
            let _ = d.get_double();
            let _ = d.get_size();
        }
    }
}
