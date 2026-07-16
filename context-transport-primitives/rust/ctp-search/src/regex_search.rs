// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).

//! Regex-queryable string index with a trigram prefilter.
//!
//! Rust port of `ctp::search::RegexSearchEngine<ValueT>` from
//! `include/clio_ctp/search/regex_search_engine.h` (the index behind CTE
//! `TagQuery`, issue #598).
//!
//! Keys are indexed by their overlapping character trigrams in an inverted
//! index `trigram -> {keys}`. [`RegexSearchEngine::search`] derives the set of
//! trigrams that EVERY match of the pattern must contain, intersects their
//! posting lists to obtain a small candidate set, and then verifies each
//! candidate with the full regex. This is the technique behind Google Code
//! Search (Cox, "Regular Expression Matching with a Trigram Index").
//!
//! Correctness: the trigram extractor is CONSERVATIVE — it only treats a
//! trigram as "required" when it provably appears in every match (it comes from
//! a literal run that is neither optional nor inside an alternation/group).
//! Whenever it cannot prove a useful required trigram (alternation, groups,
//! short literals, ...) it falls back to scanning all keys. The candidate set is
//! therefore always a superset of the true matches, and the final regex
//! verification makes the result exact regardless of how good the prefilter was.
//!
//! # C++ → Rust name mapping
//!
//! | C++ (`ctp::search`) | Rust (this module) |
//! |---|---|
//! | `RegexSearchEngine<ValueT>` | [`RegexSearchEngine<V, C>`] (`C` = regex backend) |
//! | `RegexSearchEngine()` (default ctor) | [`RegexSearchEngine::new`] / `Default` (when `C: Default`) |
//! | `bool Insert(key, value)` | [`RegexSearchEngine::insert`] |
//! | `bool Delete(key)` | [`RegexSearchEngine::delete`] (also [`delete_value`](RegexSearchEngine::delete_value)) |
//! | `bool Rename(old_key, new_key)` | [`RegexSearchEngine::rename`] |
//! | `bool Contains(key)` | [`RegexSearchEngine::contains`] |
//! | `const ValueT *Find(key)` | [`RegexSearchEngine::find`] (clones) / [`with_value`](RegexSearchEngine::with_value) (borrows) |
//! | `size_t Size()` | [`RegexSearchEngine::len`] |
//! | `bool Empty()` | [`RegexSearchEngine::is_empty`] |
//! | `void Clear()` | [`RegexSearchEngine::clear`] |
//! | `SearchResult Search(pattern)` | [`RegexSearchEngine::search`] → `Result<SearchResult, RegexError>` |
//! | (n/a) | [`RegexSearchEngine::search_with`] (caller-compiled matcher) |
//! | `SearchResult` | [`SearchResult`] |
//! | `SearchResult::keys()` | [`SearchResult::keys`] |
//! | `SearchResult::size()` / `empty()` | [`SearchResult::len`] / [`SearchResult::is_empty`] |
//! | `SearchResult::begin()/end()` | [`SearchResult::iter`] |
//! | `static void Trigrams(s, out)` | `trigrams(s) -> Vec<Trigram>` (private) |
//! | `static bool ExtractRequiredTrigrams(p, out)` | `extract_required_trigrams(p) -> Option<Vec<Trigram>>` (private) |
//! | `std::regex` / `std::regex_match` | [`RegexMatcher`] + [`RegexCompiler`] traits (caller-supplied) |
//! | `std::regex_error` | [`RegexError`] |
//! | `std::shared_mutex mtx_` | `std::sync::RwLock<Inner<V>>` |
//! | `unordered_map<string, ValueT> entries_` | `HashMap<Arc<str>, V>` |
//! | `unordered_map<string, unordered_set<const string*>> index_` | `HashMap<[u8; 3], HashSet<Arc<str>>>` |
//!
//! # Semantic divergences from the C++ original
//!
//! 1. **No regex engine ships with this port (the headline divergence).** The
//!    crate may not depend on the `regex` crate, and std has no regex. The
//!    trigram prefilter and the index are ported faithfully, but the *match
//!    step* is injected by the caller through two small traits:
//!    [`RegexCompiler`] (pattern text → matcher, mirroring the `std::regex`
//!    constructor, including its "throws on a bad pattern" behaviour, as
//!    `Result`) and [`RegexMatcher`] (`is_match`, mirroring `std::regex_match`
//!    — whole-key match). Consequently **the grammar is whatever the caller's
//!    backend implements**; the C++ engine hardcodes ECMAScript. The prefilter
//!    below still parses ECMAScript-flavoured syntax exactly as the C++ does, so
//!    a backend with a different grammar could in principle disagree with the
//!    prefilter's assumptions and lose matches. Callers must supply an
//!    ECMAScript-compatible backend (e.g. the `regex` crate with the pattern
//!    anchored, or a `std::regex` FFI shim) for byte-for-byte parity.
//! 2. **`Find` cannot hand out a live reference.** C++ returns
//!    `const ValueT *` into the map (explicitly unstable under concurrent
//!    mutation). Rust cannot let a reference escape the `RwLock` guard, so
//!    [`find`](RegexSearchEngine::find) returns a clone (`V: Clone`) and
//!    [`with_value`](RegexSearchEngine::with_value) offers a zero-copy borrow
//!    scoped to a closure that runs under the read lock.
//! 3. **`SearchResult` iteration.** C++ dereferences `*Find(k)` per element —
//!    undefined behaviour (null deref) if the key was deleted after the
//!    snapshot. [`SearchResult::iter`] instead yields `(&str, V)` with the value
//!    cloned under the read lock, and **skips** keys that vanished since the
//!    snapshot. Under the C++-documented contract (no concurrent mutation while
//!    iterating) the behaviour is identical. [`SearchResult::keys`] is the
//!    concurrency-safe accessor in both.
//! 4. **Postings hold `Arc<str>`, not raw `const std::string*`.** The C++ stores
//!    a bare pointer to the single key copy owned by `entries_` (so long keys are
//!    not copied into every posting) and relies on node-pointer stability, with
//!    `Delete` scrubbing the postings before erasing the node. `Arc<str>` keeps
//!    that "don't copy the key" property with zero `unsafe` and no lifetime
//!    coupling; postings therefore hash/compare keys **by content** instead of by
//!    pointer identity. Since `entries_` holds each key exactly once, content
//!    identity and pointer identity coincide — set semantics are unchanged. The
//!    C++ scrub-before-erase ordering is likewise unnecessary and not reproduced.
//! 5. **Trigrams are `[u8; 3]`, not `std::string`.** Both are byte-oriented:
//!    trigrams are cut at byte offsets, so multi-byte UTF-8 is split mid-scalar
//!    exactly as in C++ (`std::string::substr`). Keys/patterns remain `&str`
//!    (valid UTF-8), a Rust restriction; C++ accepts arbitrary bytes.
//! 6. **`{n,m}` minimum-count overflow.** C++ uses `std::atol`, whose overflow
//!    behaviour is undefined. This port saturates to `u64::MAX` — i.e. a huge
//!    minimum count stays non-zero, so the preceding atom is kept in the literal
//!    run (never over-constraining the filter).
//! 7. **`Search` returns `Result` instead of throwing `std::regex_error`.**
//!    Panics from a caller's backend are not caught.
//! 8. **Lock poisoning is ignored** (`into_inner` on a poisoned `RwLock`) because
//!    C++ `shared_mutex` has no such concept. Rust's `RwLock` also makes no
//!    reader/writer fairness guarantee, as `std::shared_mutex` does not; the
//!    #680 mitigation is preserved regardless — compiling the pattern and running
//!    the matcher both happen OUTSIDE the lock, which is what kept writers from
//!    starving.
//! 9. **No default backend is provided**, so the engine is unusable until a
//!    caller supplies one — see divergence 1. Tests here drive a small
//!    purpose-built backtracking matcher.
//! 10. `insert` takes `&str` + `V` by value (Rust move semantics) where C++ takes
//!     `const std::string&` + `const ValueT&` and copies.
//!
//! # Thread-safety
//!
//! Internally synchronized with an `RwLock`, mirroring the C++ `shared_mutex`:
//! mutators (`insert`/`delete`/`rename`/`clear`) take it exclusively; queries
//! (`search`/`contains`/`find`/`len`/`is_empty`) take it shared. `search`
//! returns a SNAPSHOT of the matching keys.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// A trigram: three consecutive **bytes** of a key (C++: a 3-char `std::string`).
pub type Trigram = [u8; 3];

// ---------------------------------------------------------------------------
// Regex backend (divergence 1)
// ---------------------------------------------------------------------------

/// Error reported when a pattern cannot be compiled.
///
/// Stands in for `std::regex_error`, which the C++ `Search` lets propagate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegexError {
    /// The pattern that failed to compile.
    pub pattern: String,
    /// Backend-specific explanation.
    pub message: String,
}

impl RegexError {
    /// Build an error for `pattern` with the backend's `message`.
    pub fn new(pattern: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            pattern: pattern.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for RegexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid regex `{}`: {}", self.pattern, self.message)
    }
}

impl std::error::Error for RegexError {}

/// A compiled regex: the port's stand-in for `std::regex_match(key, re)`.
///
/// `is_match` must have **whole-string** semantics (C++ `regex_match`, not
/// `regex_search`): it returns `true` only when the ENTIRE `text` matches.
pub trait RegexMatcher {
    /// True when the whole of `text` matches the pattern.
    fn is_match(&self, text: &str) -> bool;
}

/// Any `Fn(&str) -> bool` is a matcher (covers [`BoxedMatcher`] too).
impl<F: Fn(&str) -> bool> RegexMatcher for F {
    fn is_match(&self, text: &str) -> bool {
        self(text)
    }
}

/// Compiles pattern text into a [`RegexMatcher`]; stands in for the `std::regex`
/// constructor (which throws `std::regex_error` on a bad pattern).
pub trait RegexCompiler {
    /// The compiled representation produced by this backend.
    type Matcher: RegexMatcher;
    /// Compile `pattern`, or report why it is invalid.
    fn compile(&self, pattern: &str) -> Result<Self::Matcher, RegexError>;
}

/// A type-erased matcher, for backends that would rather return a closure.
pub type BoxedMatcher = Box<dyn Fn(&str) -> bool + Send + Sync>;

/// Adapts a `Fn(&str) -> Result<BoxedMatcher, RegexError>` into a
/// [`RegexCompiler`], so a backend can be supplied as a closure.
pub struct FnCompiler<F>(pub F);

impl<F> RegexCompiler for FnCompiler<F>
where
    F: Fn(&str) -> Result<BoxedMatcher, RegexError>,
{
    type Matcher = BoxedMatcher;
    fn compile(&self, pattern: &str) -> Result<BoxedMatcher, RegexError> {
        (self.0)(pattern)
    }
}

impl<F: fmt::Debug> fmt::Debug for FnCompiler<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("FnCompiler").field(&self.0).finish()
    }
}

// ---------------------------------------------------------------------------
// Trigram extraction (ports the private statics)
// ---------------------------------------------------------------------------

/// The unique overlapping trigrams of `s`, sorted (empty when `s` is < 3 bytes).
///
/// Ports `static void Trigrams(const std::string &s, std::vector<std::string> &out)`.
fn trigrams(s: &str) -> Vec<Trigram> {
    let b = s.as_bytes();
    if b.len() < 3 {
        return Vec::new();
    }
    // C++ collects into a std::set: unique + sorted.
    let uniq: BTreeSet<Trigram> = b.windows(3).map(|w| [w[0], w[1], w[2]]).collect();
    uniq.into_iter().collect()
}

/// C++ `flush()` lambda: keep the run only if it can yield a trigram.
fn flush_run(run: &mut Vec<u8>, runs: &mut Vec<Vec<u8>>) {
    if run.len() >= 3 {
        runs.push(std::mem::take(run));
    } else {
        run.clear();
    }
}

/// Derive the trigrams that EVERY match of `p` must contain (AND semantics).
///
/// Ports `static bool ExtractRequiredTrigrams(const std::string &p, std::vector<std::string> &out)`
/// with the `bool`/out-param pair folded into an `Option`:
///
/// * `None` — the C++ `return false`: a construct the extractor cannot reason
///   about safely (alternation `|`, groups `()`, a dangling escape). The caller
///   must scan all keys.
/// * `Some(v)` — the C++ `return true`. An EMPTY `v` means "no useful
///   constraint, scan all". The function NEVER emits a trigram that is not
///   guaranteed to appear in every match.
fn extract_required_trigrams(p: &str) -> Option<Vec<Trigram>> {
    let p = p.as_bytes();
    let n = p.len();
    let mut runs: Vec<Vec<u8>> = Vec::new();
    let mut run: Vec<u8> = Vec::new();

    let mut i = 0usize;
    while i < n {
        let c = p[i];
        match c {
            // Alternation / groups: a literal here is not guaranteed in every
            // match. Bail to a full scan.
            b'|' | b'(' | b')' => return None,
            b'\\' => {
                if i + 1 >= n {
                    return None; // dangling escape -> bail
                }
                let nx = p[i + 1];
                if nx.is_ascii_alphanumeric() {
                    // \d \w \s \b ... a class, not a reliable literal.
                    flush_run(&mut run, &mut runs);
                } else {
                    run.push(nx); // \. \/ \+ \\ ... a literal character
                }
                i += 2;
            }
            b'[' => {
                // Character class matches one of a set -> not a fixed literal.
                flush_run(&mut run, &mut runs);
                i += 1;
                if i < n && p[i] == b'^' {
                    i += 1;
                }
                if i < n && p[i] == b']' {
                    i += 1; // a leading ']' is a literal member of the class
                }
                while i < n && p[i] != b']' {
                    if p[i] == b'\\' && i + 1 < n {
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                if i < n {
                    i += 1; // consume ']'
                }
            }
            b'.' | b'^' | b'$' => {
                flush_run(&mut run, &mut runs);
                i += 1;
            }
            b'*' | b'?' => {
                // The preceding atom is optional: drop it and break the run.
                run.pop();
                flush_run(&mut run, &mut runs);
                i += 1;
            }
            b'+' => {
                // The preceding atom occurs >=1 times: it stays in the run, but
                // the run cannot extend across the repetition.
                flush_run(&mut run, &mut runs);
                i += 1;
            }
            b'{' => {
                // Possible {n}, {n,}, {n,m} quantifier. Parse the minimum count.
                let mut j = i + 1;
                let digits_start = j;
                while j < n && p[j].is_ascii_digit() {
                    j += 1;
                }
                let mn = &p[digits_start..j];
                let mut k = j;
                while k < n && p[k] != b'}' {
                    k += 1;
                }
                // Short-circuit order matches C++: p[j] is only read when k < n,
                // which implies j < n.
                let is_quant = !mn.is_empty() && k < n && (p[j] == b'}' || p[j] == b',');
                if !is_quant {
                    run.push(b'{'); // a literal '{'
                    i += 1;
                    continue;
                }
                // C++ std::atol; overflow saturates here instead of being UB
                // (divergence 6). `mn` is ASCII digits, so from_utf8 cannot fail.
                let min_count: u64 = std::str::from_utf8(mn)
                    .ok()
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(u64::MAX);
                if min_count == 0 {
                    run.pop(); // preceding atom optional (no-op on an empty run)
                }
                flush_run(&mut run, &mut runs);
                i = k + 1;
            }
            _ => {
                run.push(c);
                i += 1;
            }
        }
    }
    flush_run(&mut run, &mut runs);

    let mut tri: BTreeSet<Trigram> = BTreeSet::new();
    for r in &runs {
        for w in r.windows(3) {
            tri.insert([w[0], w[1], w[2]]);
        }
    }
    Some(tri.into_iter().collect())
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// The lock-guarded state: `entries_` + `index_` in the C++.
struct Inner<V> {
    /// key -> value. Owns the single `Arc<str>` copy of each key.
    entries: HashMap<Arc<str>, V>,
    /// trigram -> set of keys. Shares the key buffers with `entries` (see
    /// divergence 4) to keep the index small for long keys.
    index: HashMap<Trigram, HashSet<Arc<str>>>,
}

impl<V> Inner<V> {
    /// Ports `InsertLocked`.
    fn insert(&mut self, key: &str, value: V) -> bool {
        // C++ insert_or_assign: overwrite in place, trigrams already indexed.
        if let Some(slot) = self.entries.get_mut(key) {
            *slot = value;
            return false;
        }
        let kp: Arc<str> = Arc::from(key);
        self.entries.insert(Arc::clone(&kp), value);
        for t in trigrams(key) {
            self.index.entry(t).or_default().insert(Arc::clone(&kp));
        }
        true
    }

    /// Ports `DeleteLocked`, additionally handing back the removed value (which
    /// `Rename` needs; C++ moves it out of the node before calling this).
    fn delete(&mut self, key: &str) -> Option<V> {
        let value = self.entries.remove(key)?;
        for t in trigrams(key) {
            if let Some(posting) = self.index.get_mut(&t) {
                posting.remove(key);
                if posting.is_empty() {
                    self.index.remove(&t);
                }
            }
        }
        Some(value)
    }
}

/// A regex-queryable string index that maps string keys to user-defined values.
///
/// `V` is the C++ `ValueT`; `C` is the caller-supplied regex backend
/// (divergence 1) — a [`RegexCompiler`], typically a unit struct wrapping a real
/// regex library, or a [`FnCompiler`] closure.
///
/// See the [module docs](self) for the algorithm, thread-safety, and the full
/// list of divergences.
pub struct RegexSearchEngine<V, C> {
    inner: RwLock<Inner<V>>,
    compiler: C,
}

impl<V, C: Default> Default for RegexSearchEngine<V, C> {
    /// C++ `RegexSearchEngine() = default;` — available when the backend is
    /// default-constructible.
    fn default() -> Self {
        Self::new(C::default())
    }
}

impl<V, C> RegexSearchEngine<V, C> {
    /// Create an empty engine that verifies candidates with `compiler`.
    pub fn new(compiler: C) -> Self {
        Self {
            inner: RwLock::new(Inner {
                entries: HashMap::new(),
                index: HashMap::new(),
            }),
            compiler,
        }
    }

    /// The regex backend this engine was built with.
    pub fn compiler(&self) -> &C {
        &self.compiler
    }

    /// Shared lock. Poisoning is ignored: C++ `shared_mutex` has no such notion
    /// and the guarded state stays structurally valid on panic (divergence 8).
    fn read(&self) -> RwLockReadGuard<'_, Inner<V>> {
        self.inner.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Exclusive lock; see [`read`](Self::read) about poisoning.
    fn write(&self) -> RwLockWriteGuard<'_, Inner<V>> {
        self.inner.write().unwrap_or_else(|e| e.into_inner())
    }

    /// Bind `value` to `key`. If `key` already exists its value is overwritten.
    ///
    /// Returns `true` if a new key was added, `false` if an existing one was
    /// updated. Ports `Insert`.
    pub fn insert(&self, key: &str, value: V) -> bool {
        self.write().insert(key, value)
    }

    /// Remove `key`. Returns `true` if it existed. Ports `Delete`.
    pub fn delete(&self, key: &str) -> bool {
        self.write().delete(key).is_some()
    }

    /// Like [`delete`](Self::delete) but hands back the removed value.
    ///
    /// Rust-only convenience (C++ drops the value with the map node).
    pub fn delete_value(&self, key: &str) -> Option<V> {
        self.write().delete(key)
    }

    /// Move the entry at `old_key` to `new_key`, preserving its value. If
    /// `new_key` already exists its value is overwritten by the moved one.
    ///
    /// Returns `false` if `old_key` does not exist. Ports `Rename`.
    pub fn rename(&self, old_key: &str, new_key: &str) -> bool {
        let mut g = self.write();
        if !g.entries.contains_key(old_key) {
            return false;
        }
        if old_key == new_key {
            return true;
        }
        // C++ moves the value out, then DeleteLocked + InsertLocked.
        let Some(moved) = g.delete(old_key) else {
            return false; // unreachable: presence checked under the same lock
        };
        g.insert(new_key, moved);
        true
    }

    /// Ports `Contains`.
    pub fn contains(&self, key: &str) -> bool {
        self.read().entries.contains_key(key)
    }

    /// Run `f` on the value bound to `key` under the read lock; `None` if absent.
    ///
    /// The zero-copy analogue of C++ `Find` (divergence 2). Do not call back
    /// into the engine from `f` — a mutator would deadlock.
    pub fn with_value<R>(&self, key: &str, f: impl FnOnce(&V) -> R) -> Option<R> {
        self.read().entries.get(key).map(f)
    }

    /// Number of indexed keys. Ports `Size`.
    pub fn len(&self) -> usize {
        self.read().entries.len()
    }

    /// Ports `Empty`.
    pub fn is_empty(&self) -> bool {
        self.read().entries.is_empty()
    }

    /// Drop every entry and the whole trigram index. Ports `Clear`.
    pub fn clear(&self) {
        let mut g = self.write();
        g.entries.clear();
        g.index.clear();
    }

    /// Every (key, value) whose key fully matches `pattern`, verified with a
    /// matcher the caller compiled themselves.
    ///
    /// `pattern` is still used for the trigram prefilter, so it MUST be the same
    /// pattern `matcher` was compiled from — otherwise the prefilter may discard
    /// true matches. This is the infallible half of [`search`](Self::search),
    /// for callers that cache compiled regexes.
    pub fn search_with<M: RegexMatcher + ?Sized>(
        &self,
        pattern: &str,
        matcher: &M,
    ) -> SearchResult<'_, V, C> {
        let required = extract_required_trigrams(pattern);

        // Snapshot the candidate keys under the shared lock, then RELEASE it and
        // run the (expensive) matcher over the snapshot. Cloning an Arc is far
        // cheaper than matching, so the lock is held briefly and writers do not
        // starve (#680). The snapshot also makes SearchResult stable under
        // concurrent mutation.
        let candidates: Vec<Arc<str>> = {
            let g = self.read();
            match required {
                // Candidates = keys present in the posting lists of ALL required
                // trigrams. Walk the smallest posting list and test membership in
                // the others; this is a superset of the true matches.
                Some(ref req) if !req.is_empty() => {
                    let mut postings: Vec<&HashSet<Arc<str>>> = Vec::with_capacity(req.len());
                    for t in req {
                        match g.index.get(t) {
                            // Some required trigram indexes no key => no match.
                            None => return SearchResult::new(Vec::new(), self),
                            Some(posting) => postings.push(posting),
                        }
                    }
                    // `min_by_key` keeps the first minimum, like the C++ `<` scan.
                    let smallest = *postings
                        .iter()
                        .min_by_key(|p| p.len())
                        .expect("req is non-empty");
                    let mut out = Vec::new();
                    for kp in smallest.iter() {
                        let k: &str = kp;
                        if postings.iter().all(|p| p.contains(k)) {
                            out.push(Arc::clone(kp)); // snapshot under the lock
                        }
                    }
                    out
                }
                // No usable prefilter: every key is a candidate.
                _ => g.entries.keys().cloned().collect(),
            }
        }; // shared lock released before matching

        let mut matches: Vec<String> = candidates
            .into_iter()
            .filter(|c| matcher.is_match(c))
            .map(|c| c.to_string())
            .collect();
        matches.sort();
        SearchResult::new(matches, self)
    }
}

impl<V, C: RegexCompiler> RegexSearchEngine<V, C> {
    /// Every (key, value) whose key fully matches `pattern`.
    ///
    /// Ports `Search`; the C++ `throws std::regex_error` becomes `Err`
    /// (divergence 7). Like the C++, the pattern is compiled OUTSIDE the lock —
    /// compilation touches no shared state and holding the shared lock across it
    /// starved writers (#680).
    pub fn search(&self, pattern: &str) -> Result<SearchResult<'_, V, C>, RegexError> {
        let matcher = self.compiler.compile(pattern)?;
        Ok(self.search_with(pattern, &matcher))
    }
}

impl<V: fmt::Debug, C> fmt::Debug for RegexSearchEngine<V, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let g = self.read();
        f.debug_struct("RegexSearchEngine")
            .field("entries", &g.entries)
            .field("trigrams", &g.index.len())
            .finish()
    }
}

/// Iterable result of a [`search`](RegexSearchEngine::search).
///
/// Holds a snapshot of the matching keys (sorted); bound values are fetched live
/// from the engine on iteration, so the engine must outlive the result. Ports
/// `RegexSearchEngine::SearchResult`.
pub struct SearchResult<'a, V, C> {
    keys: Vec<String>,
    eng: &'a RegexSearchEngine<V, C>,
}

impl<'a, V, C> SearchResult<'a, V, C> {
    fn new(keys: Vec<String>, eng: &'a RegexSearchEngine<V, C>) -> Self {
        Self { keys, eng }
    }

    /// The matching keys (sorted), e.g. for callers that only need names.
    ///
    /// This is the concurrency-safe accessor: a plain snapshot.
    pub fn keys(&self) -> &[String] {
        &self.keys
    }

    /// Number of matching keys. Ports `SearchResult::size()`.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Ports `SearchResult::empty()`.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// The keys, consumed.
    pub fn into_keys(self) -> Vec<String> {
        self.keys
    }
}

impl<V, C> fmt::Debug for SearchResult<'_, V, C> {
    /// Shows the key snapshot only; values are not held by the result.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SearchResult")
            .field("keys", &self.keys)
            .finish()
    }
}

impl<V: Clone, C> SearchResult<'_, V, C> {
    /// Yield `(key, value)` pairs, fetching each value live from the engine.
    ///
    /// Ports `SearchResult::begin()/end()`. Divergence 3: values are CLONED
    /// under the engine's read lock (Rust cannot let a reference escape the
    /// guard), and keys deleted since the snapshot are SKIPPED rather than
    /// null-dereferenced.
    pub fn iter(&self) -> impl Iterator<Item = (&str, V)> + '_ {
        self.keys
            .iter()
            .filter_map(|k| self.eng.find(k).map(|v| (k.as_str(), v)))
    }
}

impl<V: Clone, C> RegexSearchEngine<V, C> {
    /// The value bound to `key`, cloned, or `None` if absent.
    ///
    /// Closest safe analogue of C++ `const ValueT *Find(key)` (divergence 2);
    /// see [`with_value`](Self::with_value) to avoid the clone.
    pub fn find(&self, key: &str) -> Option<V> {
        self.read().entries.get(key).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Barrier;

    // -----------------------------------------------------------------------
    // A tiny backtracking regex backend, TEST-ONLY (divergence 9).
    //
    // Supports the constructs the prefilter reasons about: literals, `.`,
    // char classes (with `^` negation and `a-z` ranges), `*` `+` `?` `{n,m}`,
    // top-level `|`, and `\` escapes (`\d \w \s` + escaped literals). `^`/`$`
    // are accepted and ignored, since matching is whole-string anyway (C++
    // regex_match). Deliberately NOT a general ECMAScript engine — groups are
    // rejected as invalid patterns.
    // -----------------------------------------------------------------------

    #[derive(Debug, Clone)]
    enum Node {
        Lit(u8),
        Any,
        Class { neg: bool, items: Vec<(u8, u8)> },
    }

    impl Node {
        fn matches(&self, b: u8) -> bool {
            match self {
                Node::Lit(l) => *l == b,
                Node::Any => true,
                Node::Class { neg, items } => {
                    let hit = items.iter().any(|(lo, hi)| b >= *lo && b <= *hi);
                    hit != *neg
                }
            }
        }
    }

    #[derive(Debug, Clone)]
    struct Piece {
        node: Node,
        min: u32,
        max: u32,
    }

    #[derive(Debug, Clone)]
    struct MiniRegex {
        alts: Vec<Vec<Piece>>,
    }

    fn class_items_for(esc: u8) -> Option<(bool, Vec<(u8, u8)>)> {
        match esc {
            b'd' => Some((false, vec![(b'0', b'9')])),
            b'D' => Some((true, vec![(b'0', b'9')])),
            b'w' => Some((
                false,
                vec![(b'0', b'9'), (b'A', b'Z'), (b'a', b'z'), (b'_', b'_')],
            )),
            b's' => Some((false, vec![(b' ', b' '), (b'\t', b'\t'), (b'\n', b'\n')])),
            _ => None,
        }
    }

    fn parse_alt(p: &[u8], pattern: &str) -> Result<Vec<Piece>, RegexError> {
        let n = p.len();
        let mut out: Vec<Piece> = Vec::new();
        let mut i = 0usize;
        while i < n {
            let node = match p[i] {
                b'(' | b')' => {
                    return Err(RegexError::new(pattern, "groups unsupported by MiniRegex"))
                }
                b'^' | b'$' => {
                    i += 1;
                    continue;
                }
                b'.' => {
                    i += 1;
                    Node::Any
                }
                b'\\' => {
                    if i + 1 >= n {
                        return Err(RegexError::new(pattern, "dangling escape"));
                    }
                    let nx = p[i + 1];
                    i += 2;
                    match class_items_for(nx) {
                        Some((neg, items)) => Node::Class { neg, items },
                        None if nx.is_ascii_alphanumeric() => {
                            return Err(RegexError::new(pattern, "unsupported escape"))
                        }
                        None => Node::Lit(nx),
                    }
                }
                b'[' => {
                    i += 1;
                    let mut neg = false;
                    if i < n && p[i] == b'^' {
                        neg = true;
                        i += 1;
                    }
                    let mut items: Vec<(u8, u8)> = Vec::new();
                    if i < n && p[i] == b']' {
                        items.push((b']', b']'));
                        i += 1;
                    }
                    while i < n && p[i] != b']' {
                        let lo = if p[i] == b'\\' && i + 1 < n {
                            i += 2;
                            p[i - 1]
                        } else {
                            i += 1;
                            p[i - 1]
                        };
                        if i + 1 < n && p[i] == b'-' && p[i + 1] != b']' {
                            items.push((lo, p[i + 1]));
                            i += 2;
                        } else {
                            items.push((lo, lo));
                        }
                    }
                    if i >= n {
                        return Err(RegexError::new(pattern, "unterminated class"));
                    }
                    i += 1;
                    Node::Class { neg, items }
                }
                b'*' | b'+' | b'?' | b'{' => {
                    return Err(RegexError::new(pattern, "quantifier without atom"))
                }
                c => {
                    i += 1;
                    Node::Lit(c)
                }
            };
            let (min, max) = if i < n {
                match p[i] {
                    b'*' => {
                        i += 1;
                        (0, u32::MAX)
                    }
                    b'+' => {
                        i += 1;
                        (1, u32::MAX)
                    }
                    b'?' => {
                        i += 1;
                        (0, 1)
                    }
                    b'{' => {
                        let close = (i..n).find(|&k| p[k] == b'}');
                        match close {
                            None => (1, 1), // literal '{' handled below
                            Some(k) => {
                                let body = std::str::from_utf8(&p[i + 1..k]).unwrap_or("");
                                let parsed = if let Some((a, b)) = body.split_once(',') {
                                    match (a.parse::<u32>(), b.trim().is_empty()) {
                                        (Ok(lo), true) => Some((lo, u32::MAX)),
                                        (Ok(lo), false) => b.parse::<u32>().ok().map(|hi| (lo, hi)),
                                        _ => None,
                                    }
                                } else {
                                    body.parse::<u32>().ok().map(|x| (x, x))
                                };
                                match parsed {
                                    Some(mm) => {
                                        i = k + 1;
                                        mm
                                    }
                                    None => (1, 1),
                                }
                            }
                        }
                    }
                    _ => (1, 1),
                }
            } else {
                (1, 1)
            };
            out.push(Piece { node, min, max });
        }
        Ok(out)
    }

    fn match_pieces(pieces: &[Piece], s: &[u8]) -> bool {
        let Some(p) = pieces.first() else {
            return s.is_empty();
        };
        let rest = &pieces[1..];
        let mut i = 0usize;
        let mut cnt = 0u32;
        while cnt < p.min {
            if i < s.len() && p.node.matches(s[i]) {
                i += 1;
                cnt += 1;
            } else {
                return false;
            }
        }
        loop {
            if match_pieces(rest, &s[i..]) {
                return true;
            }
            if cnt >= p.max || i >= s.len() || !p.node.matches(s[i]) {
                return false;
            }
            i += 1;
            cnt += 1;
        }
    }

    impl RegexMatcher for MiniRegex {
        fn is_match(&self, text: &str) -> bool {
            self.alts.iter().any(|a| match_pieces(a, text.as_bytes()))
        }
    }

    #[derive(Debug, Default, Clone, Copy)]
    struct Mini;

    impl RegexCompiler for Mini {
        type Matcher = MiniRegex;
        fn compile(&self, pattern: &str) -> Result<MiniRegex, RegexError> {
            let mut alts = Vec::new();
            for part in pattern.split('|') {
                alts.push(parse_alt(part.as_bytes(), pattern)?);
            }
            Ok(MiniRegex { alts })
        }
    }

    type Engine = RegexSearchEngine<u32, Mini>;

    fn tg(s: &str) -> Vec<Trigram> {
        trigrams(s)
    }

    fn tri(s: &str) -> Trigram {
        let b = s.as_bytes();
        [b[0], b[1], b[2]]
    }

    fn req(pattern: &str) -> Option<Vec<String>> {
        extract_required_trigrams(pattern).map(|v| {
            v.into_iter()
                .map(|t| String::from_utf8_lossy(&t).into_owned())
                .collect()
        })
    }

    // ---------------------------------------------------------------- Trigrams

    #[test]
    fn trigrams_shorter_than_three_is_empty() {
        assert!(tg("").is_empty());
        assert!(tg("a").is_empty());
        assert!(tg("ab").is_empty());
    }

    #[test]
    fn trigrams_boundary_exactly_three() {
        assert_eq!(tg("abc"), vec![tri("abc")]);
    }

    #[test]
    fn trigrams_are_overlapping_unique_and_sorted() {
        assert_eq!(tg("abcd"), vec![tri("abc"), tri("bcd")]);
        // "aaaa" -> {aaa} only: the std::set dedupes.
        assert_eq!(tg("aaaa"), vec![tri("aaa")]);
        // sorted, not insertion order
        assert_eq!(
            tg("cbaabc"),
            vec![tri("aab"), tri("abc"), tri("baa"), tri("cba")]
        );
    }

    #[test]
    fn trigrams_are_byte_oriented_like_cpp() {
        // "é" is 2 bytes: "éx" is 3 bytes -> exactly one (split-mid-scalar) trigram.
        let t = tg("éx");
        assert_eq!(t.len(), 1);
        assert_eq!(t[0], [0xC3, 0xA9, b'x']);
    }

    // ------------------------------------------------ ExtractRequiredTrigrams

    #[test]
    fn required_plain_literal() {
        assert_eq!(
            req("hello"),
            Some(vec!["ell".into(), "hel".into(), "llo".into()])
        );
    }

    #[test]
    fn required_empty_pattern_is_prefilterable_but_unconstrained() {
        assert_eq!(req(""), Some(vec![]));
    }

    #[test]
    fn required_bails_on_alternation_and_groups() {
        assert_eq!(req("foo|bar"), None);
        assert_eq!(req("(abc)"), None);
        assert_eq!(req("abc)"), None);
        assert_eq!(req("a(bc"), None);
    }

    #[test]
    fn required_bails_on_dangling_escape() {
        assert_eq!(req("abc\\"), None);
        assert_eq!(req("\\"), None);
    }

    #[test]
    fn required_escape_class_breaks_run_escaped_literal_extends_it() {
        // \d is a class -> flush "abcd"
        assert_eq!(req("abcd\\d"), Some(vec!["abc".into(), "bcd".into()]));
        // \. is a literal '.' -> stays in the run
        assert_eq!(
            req("ab\\.cd"),
            Some(
                vec!["ab.".into(), "b.c".into(), ".cd".into()]
                    .into_iter()
                    .collect::<BTreeSet<String>>()
                    .into_iter()
                    .collect::<Vec<_>>()
            )
        );
        // escaped backslash is a literal (run "a\bc"; '\' = 0x5C sorts before 'a')
        assert_eq!(req("a\\\\bc"), Some(vec!["\\bc".into(), "a\\b".into()]));
    }

    #[test]
    fn required_char_class_breaks_the_run() {
        assert_eq!(req("[abc]def"), Some(vec!["def".into()]));
        assert_eq!(req("[^a-z]xyz"), Some(vec!["xyz".into()]));
        // escaped ']' inside the class must not terminate it early
        assert_eq!(req("[a\\]b]xyz"), Some(vec!["xyz".into()]));
        // a LEADING ']' is a literal member of the class
        assert_eq!(req("[]]abc"), Some(vec!["abc".into()]));
        // unterminated class swallows the rest -> no runs
        assert_eq!(req("abc[def"), Some(vec!["abc".into()]));
    }

    #[test]
    fn required_dot_and_anchors_break_the_run() {
        assert_eq!(req("abc.def"), Some(vec!["abc".into(), "def".into()]));
        assert_eq!(req("^abcd$"), Some(vec!["abc".into(), "bcd".into()]));
    }

    #[test]
    fn required_star_and_question_drop_the_preceding_atom() {
        // '*' pops 'd', leaving "abc"
        assert_eq!(req("abcd*"), Some(vec!["abc".into()]));
        assert_eq!(req("abcd?ef"), Some(vec!["abc".into()]));
        // run too short after the pop -> no constraint (full scan), still Some
        assert_eq!(req("abc*"), Some(vec![]));
        // pop on an empty run must not underflow
        assert_eq!(req("*"), Some(vec![]));
        assert_eq!(req("?abc"), Some(vec!["abc".into()]));
    }

    #[test]
    fn required_plus_keeps_the_atom_but_ends_the_run() {
        // 'd' occurs >=1 times: kept. Run "abcd" flushed, cannot extend over 'e'.
        assert_eq!(req("abcd+e"), Some(vec!["abc".into(), "bcd".into()]));
    }

    #[test]
    fn required_brace_quantifier_min_nonzero_keeps_atom() {
        assert_eq!(req("abc{2}def"), Some(vec!["abc".into(), "def".into()]));
        assert_eq!(req("abc{2,}def"), Some(vec!["abc".into(), "def".into()]));
        assert_eq!(req("abc{2,5}def"), Some(vec!["abc".into(), "def".into()]));
    }

    #[test]
    fn required_brace_quantifier_min_zero_drops_atom() {
        // {0,2} makes 'd' optional -> pop it, run "abc"
        assert_eq!(req("abcd{0,2}ef"), Some(vec!["abc".into()]));
        assert_eq!(req("abcd{0}"), Some(vec!["abc".into()]));
        // leading zeros still parse as 0
        assert_eq!(req("abcd{00,2}"), Some(vec!["abc".into()]));
    }

    #[test]
    fn required_brace_min_count_overflow_saturates_and_keeps_atom() {
        // Divergence 6: C++ std::atol would overflow (UB); we saturate to a
        // non-zero count, so 'd' is KEPT.
        let huge = "abcd{99999999999999999999999}";
        assert_eq!(req(huge), Some(vec!["abc".into(), "bcd".into()]));
    }

    #[test]
    fn required_non_quantifier_brace_is_a_literal() {
        // no digits -> '{' is a literal, '}' falls through to the default arm.
        // Run is "a{abc}"; sorted bytes put "abc" before "a{a" ('b' < '{').
        assert_eq!(
            req("a{abc}"),
            Some(vec!["abc".into(), "a{a".into(), "bc}".into(), "{ab".into()])
        );
        // unterminated brace -> literal too: the whole "ab{12" stays one run
        assert_eq!(
            req("ab{12"),
            Some(vec!["ab{".into(), "b{1".into(), "{12".into()])
        );
        // digits but no '}' at all
        assert_eq!(
            req("abcd{3"),
            Some(vec!["abc".into(), "bcd".into(), "cd{".into(), "d{3".into()])
        );
    }

    #[test]
    fn required_lone_quantifier_yields_no_constraint() {
        assert_eq!(req("{2}"), Some(vec![]));
        assert_eq!(req("\\d{3}"), Some(vec![]));
    }

    #[test]
    fn required_realistic_patterns() {
        // A CTE-ish tag query. The class breaks the literal into exactly two
        // runs -- "/data/run" and "/output.h5" (`\.` is a literal dot, and the
        // '+' flushes an already-empty run) -- so the required set is the union
        // of their trigrams and NOTHING spans the `[0-9]+` (no "t/o").
        let expect: Vec<String> = ["/data/run", "/output.h5"]
            .iter()
            .flat_map(|r| r.as_bytes().windows(3))
            .map(|w| String::from_utf8(w.to_vec()).unwrap())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        assert_eq!(req("/data/run[0-9]+/output\\.h5"), Some(expect));
    }

    // ---------------------------------------------------------- Insert/Delete

    #[test]
    fn insert_reports_new_vs_update_and_overwrites_value() {
        let e = Engine::default();
        assert!(e.insert("alpha", 1));
        assert!(!e.insert("alpha", 2)); // update, not new
        assert_eq!(e.len(), 1);
        assert_eq!(e.find("alpha"), Some(2));
    }

    #[test]
    fn empty_engine_state() {
        let e = Engine::default();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
        assert!(!e.contains("nope"));
        assert_eq!(e.find("nope"), None);
        assert!(!e.delete("nope"));
        assert!(e.search(".*").unwrap().is_empty());
    }

    #[test]
    fn delete_removes_and_reports() {
        let e = Engine::default();
        e.insert("alpha", 1);
        assert!(e.delete("alpha"));
        assert!(!e.delete("alpha")); // already gone
        assert!(e.is_empty());
        assert!(!e.contains("alpha"));
    }

    #[test]
    fn delete_scrubs_postings_and_prunes_empty_ones() {
        let e = Engine::default();
        e.insert("alpha", 1);
        e.insert("alphabet", 2);
        {
            let g = e.read();
            // "alp" is shared by both keys.
            assert_eq!(g.index.get(&tri("alp")).unwrap().len(), 2);
            // "abe" belongs to "alphabet" only.
            assert_eq!(g.index.get(&tri("abe")).unwrap().len(), 1);
        }
        e.delete("alphabet");
        let g = e.read();
        assert_eq!(g.index.get(&tri("alp")).unwrap().len(), 1);
        assert!(!g.index.contains_key(&tri("abe"))); // empty posting pruned
    }

    #[test]
    fn deleting_every_key_empties_the_index() {
        let e = Engine::default();
        for k in ["aaa", "aab", "xyz"] {
            e.insert(k, 0);
        }
        for k in ["aaa", "aab", "xyz"] {
            assert!(e.delete(k));
        }
        let g = e.read();
        assert!(g.entries.is_empty());
        assert!(g.index.is_empty());
    }

    #[test]
    fn short_keys_are_stored_but_not_indexed() {
        let e = Engine::default();
        assert!(e.insert("ab", 7));
        assert!(e.contains("ab"));
        assert!(e.read().index.is_empty());
        // Reachable only via the full-scan path.
        assert_eq!(e.search("ab").unwrap().keys(), ["ab"]);
        assert!(e.delete("ab"));
        assert!(e.is_empty());
    }

    #[test]
    fn update_does_not_duplicate_postings() {
        let e = Engine::default();
        e.insert("alpha", 1);
        e.insert("alpha", 2);
        assert_eq!(e.read().index.get(&tri("alp")).unwrap().len(), 1);
        // ...and one delete still fully scrubs.
        e.delete("alpha");
        assert!(e.read().index.is_empty());
    }

    // ----------------------------------------------------------------- Rename

    #[test]
    fn rename_moves_value_and_reindexes() {
        let e = Engine::default();
        e.insert("alpha", 42);
        assert!(e.rename("alpha", "omega"));
        assert!(!e.contains("alpha"));
        assert_eq!(e.find("omega"), Some(42));
        assert_eq!(e.len(), 1);
        let g = e.read();
        assert!(!g.index.contains_key(&tri("alp")));
        assert!(g.index.contains_key(&tri("ome")));
    }

    #[test]
    fn rename_missing_key_is_false() {
        let e = Engine::default();
        assert!(!e.rename("nope", "other"));
        assert!(e.is_empty());
    }

    #[test]
    fn rename_to_same_key_is_a_noop_true() {
        let e = Engine::default();
        e.insert("alpha", 1);
        assert!(e.rename("alpha", "alpha"));
        assert_eq!(e.find("alpha"), Some(1));
        assert_eq!(e.len(), 1);
        assert!(e.read().index.contains_key(&tri("alp")));
    }

    #[test]
    fn rename_onto_existing_key_overwrites_it() {
        let e = Engine::default();
        e.insert("alpha", 1);
        e.insert("omega", 2);
        assert!(e.rename("alpha", "omega"));
        assert_eq!(e.len(), 1);
        assert_eq!(e.find("omega"), Some(1)); // moved value wins
        assert!(!e.contains("alpha"));
        // "omega"'s postings must survive the InsertLocked-returns-false path.
        assert_eq!(e.search("omega").unwrap().keys(), ["omega"]);
    }

    // ------------------------------------------------------------------ Clear

    #[test]
    fn clear_drops_entries_and_index() {
        let e = Engine::default();
        e.insert("alpha", 1);
        e.insert("beta", 2);
        e.clear();
        assert!(e.is_empty());
        assert_eq!(e.len(), 0);
        let g = e.read();
        assert!(g.index.is_empty());
        drop(g);
        assert!(e.search(".*").unwrap().is_empty());
        // usable again afterwards
        assert!(e.insert("gamma", 3));
        assert_eq!(e.search(".*").unwrap().keys(), ["gamma"]);
    }

    // ----------------------------------------------------------------- Search

    fn seeded() -> Engine {
        let e = Engine::default();
        for (i, k) in [
            "/data/run1/output.h5",
            "/data/run2/output.h5",
            "/data/run10/output.h5",
            "/data/run1/input.h5",
            "/logs/run1.log",
            "readme",
        ]
        .iter()
        .enumerate()
        {
            e.insert(k, i as u32);
        }
        e
    }

    #[test]
    fn search_prefiltered_exact_literal() {
        let e = seeded();
        let r = e.search("/data/run1/output\\.h5").unwrap();
        assert_eq!(r.keys(), ["/data/run1/output.h5"]);
    }

    #[test]
    fn search_results_are_sorted() {
        let e = seeded();
        let r = e.search("/data/run.*").unwrap();
        // byte-lexicographic, so run10 sorts before run1/ ('0' < '/')
        assert_eq!(
            r.keys(),
            [
                "/data/run1/input.h5",
                "/data/run1/output.h5",
                "/data/run10/output.h5",
                "/data/run2/output.h5",
            ]
        );
    }

    #[test]
    fn search_is_a_full_match_not_a_substring_match() {
        let e = seeded();
        assert!(e.search("run1").unwrap().is_empty());
        assert!(e.search("readme").unwrap().len() == 1);
    }

    #[test]
    fn search_missing_required_trigram_short_circuits_to_empty() {
        let e = seeded();
        // "zzz" indexes no key -> the C++ early return.
        assert!(e.search("/zzzzz/nothing").unwrap().is_empty());
    }

    #[test]
    fn search_falls_back_to_full_scan_on_alternation() {
        let e = seeded();
        let r = e.search("readme|/logs/run1\\.log").unwrap();
        assert_eq!(r.keys(), ["/logs/run1.log", "readme"]);
    }

    #[test]
    fn search_falls_back_when_no_run_is_long_enough() {
        let e = Engine::default();
        e.insert("ab", 1);
        e.insert("axb", 2);
        e.insert("zz", 3);
        // "a.b" has no 3-byte literal run -> unconstrained -> full scan.
        assert_eq!(extract_required_trigrams("a.b"), Some(vec![]));
        assert_eq!(e.search("a.b").unwrap().keys(), ["axb"]);
    }

    #[test]
    fn search_prefilter_intersects_all_required_trigrams() {
        let e = Engine::default();
        e.insert("foobar", 1); // has "foo" and "bar"
        e.insert("fooqux", 2); // has "foo", not "bar"
        e.insert("bazbar", 3); // has "bar", not "foo"
        let r = e.search("foo.*bar").unwrap();
        assert_eq!(r.keys(), ["foobar"]);
    }

    #[test]
    fn search_invalid_pattern_is_an_error() {
        let e = seeded();
        // MiniRegex rejects groups; the prefilter would have bailed to a scan.
        let err = e.search("(abc)").unwrap_err();
        assert_eq!(err.pattern, "(abc)");
        assert!(!err.message.is_empty());
        assert!(err.to_string().contains("(abc)"));
        // A dangling escape is rejected by both.
        assert!(e.search("abc\\").is_err());
    }

    #[test]
    fn search_result_len_empty_keys_and_into_keys() {
        let e = seeded();
        let r = e.search("/data/run1/.*\\.h5").unwrap();
        assert_eq!(r.len(), 2);
        assert!(!r.is_empty());
        assert_eq!(r.keys().len(), 2);
        assert_eq!(r.into_keys().len(), 2);
    }

    #[test]
    fn search_result_iter_yields_live_values() {
        let e = seeded();
        let r = e.search("/data/run1/.*").unwrap();
        let got: Vec<(&str, u32)> = r.iter().collect();
        assert_eq!(
            got,
            vec![("/data/run1/input.h5", 3), ("/data/run1/output.h5", 0)]
        );
        // "live": mutate, then re-iterate the SAME snapshot.
        e.insert("/data/run1/input.h5", 99);
        let got: Vec<u32> = r.iter().map(|(_, v)| v).collect();
        assert_eq!(got, vec![99, 0]);
    }

    #[test]
    fn search_result_iter_skips_keys_deleted_after_the_snapshot() {
        // Divergence 3: C++ would deref a null Find() here.
        let e = seeded();
        let r = e.search("/data/run1/.*").unwrap();
        assert_eq!(r.len(), 2);
        e.delete("/data/run1/input.h5");
        let got: Vec<&str> = r.iter().map(|(k, _)| k).collect();
        assert_eq!(got, vec!["/data/run1/output.h5"]);
        // keys() is unaffected: it is the snapshot.
        assert_eq!(r.keys().len(), 2);
    }

    #[test]
    fn search_with_uses_a_caller_compiled_matcher() {
        let e = seeded();
        let m = Mini.compile("readme").unwrap();
        assert_eq!(e.search_with("readme", &m).keys(), ["readme"]);
        // ...and a plain closure works as a matcher too.
        let closure = |s: &str| s.ends_with(".log");
        assert_eq!(e.search_with("", &closure).keys(), ["/logs/run1.log"]);
    }

    #[test]
    fn fn_compiler_closure_backend() {
        let eng: RegexSearchEngine<u32, FnCompiler<_>> =
            RegexSearchEngine::new(FnCompiler(|p: &str| -> Result<BoxedMatcher, RegexError> {
                if p.is_empty() {
                    return Err(RegexError::new(p, "empty"));
                }
                let owned = p.to_string();
                Ok(Box::new(move |s: &str| s == owned))
            }));
        eng.insert("alpha", 1);
        eng.insert("alphabet", 2);
        assert_eq!(eng.search("alpha").unwrap().keys(), ["alpha"]);
        assert!(eng.search("").is_err());
    }

    #[test]
    fn with_value_borrows_without_cloning() {
        let e: RegexSearchEngine<String, Mini> = RegexSearchEngine::default();
        e.insert("k", "value".to_string());
        assert_eq!(e.with_value("k", |v| v.len()), Some(5));
        assert_eq!(e.with_value("missing", |v| v.len()), None);
    }

    #[test]
    fn delete_value_hands_back_the_value() {
        let e = Engine::default();
        e.insert("k", 5);
        assert_eq!(e.delete_value("k"), Some(5));
        assert_eq!(e.delete_value("k"), None);
    }

    #[test]
    fn non_clone_values_are_supported() {
        // V: Clone is only required by find()/iter(); the engine itself is not.
        struct NoClone(u32);
        let e: RegexSearchEngine<NoClone, Mini> = RegexSearchEngine::default();
        e.insert("alpha", NoClone(7));
        assert_eq!(e.with_value("alpha", |v| v.0), Some(7));
        assert_eq!(e.search("alpha").unwrap().keys(), ["alpha"]);
    }

    #[test]
    fn unicode_keys_round_trip() {
        let e = Engine::default();
        e.insert("données/α", 1);
        e.insert("données/β", 2);
        assert!(e.contains("données/α"));
        let r = e.search("données/.*").unwrap();
        // '.' is byte-wise in MiniRegex, so both multi-byte tails match ".*".
        assert_eq!(r.len(), 2);
        assert!(e.delete("données/α"));
        assert_eq!(e.search("données/.*").unwrap().len(), 1);
    }

    #[test]
    fn long_keys_work() {
        let e = Engine::default();
        let long = "x".repeat(4096);
        let key = format!("/pfx/{long}/sfx");
        e.insert(&key, 1);
        assert!(e.contains(&key));
        assert_eq!(
            e.search("/pfx/x+/sfx").unwrap().keys(),
            std::slice::from_ref(&key)
        );
        assert!(e.delete(&key));
        assert!(e.read().index.is_empty());
    }

    // --------------------------------------------- Prefilter safety (superset)

    /// The core invariant: the prefilter must never lose a true match. Compare
    /// `search` against a brute-force scan with the SAME matcher.
    #[test]
    fn prefilter_never_loses_a_match() {
        let keys = [
            "alpha",
            "alphabet",
            "beta",
            "abc",
            "ab",
            "a",
            "",
            "aaa",
            "aaaa",
            "foobar",
            "fooqux",
            "bazbar",
            "/data/run1/output.h5",
            "/data/run22/output.h5",
            "readme.md",
            "read",
            "xyz",
            "a{abc}",
            "a.b",
            "a1b2c3",
            "  spaced  ",
        ];
        let patterns = [
            "",
            ".*",
            "alpha",
            "alpha.*",
            "alphabet",
            "a+",
            "a*",
            "ab?c",
            "abc",
            "[ab]+",
            "[^x]*",
            "foo.*bar",
            "foo|bar",
            "read(me)?",
            "read.*",
            "a\\{abc\\}",
            "a\\.b",
            "a\\d.\\d.\\d",
            "\\w+",
            "\\s*[a-z]+\\s*",
            "/data/run\\d+/output\\.h5",
            "x{1,3}",
            "a{0,2}bc",
            "aaa+",
            "b.ta",
            "[]]x",
            "{2}",
            "z*",
        ];
        let e = Engine::default();
        for (i, k) in keys.iter().enumerate() {
            e.insert(k, i as u32);
        }
        for p in patterns {
            let Ok(m) = Mini.compile(p) else { continue }; // MiniRegex can't do groups
            let mut expect: Vec<String> = keys
                .iter()
                .filter(|k| m.is_match(k))
                .map(|k| k.to_string())
                .collect();
            expect.sort();
            expect.dedup();
            let got = e.search(p).unwrap();
            assert_eq!(got.keys(), expect.as_slice(), "pattern `{p}` diverged");
        }
    }

    // ------------------------------------------------------------ Concurrency

    #[test]
    fn concurrent_readers_and_writers() {
        const WRITERS: usize = 4;
        const PER_WRITER: usize = 64;
        const READERS: usize = 3;

        let e = Arc::new(Engine::default());
        let barrier = Arc::new(Barrier::new(WRITERS + READERS));
        let reads = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();

        for w in 0..WRITERS {
            let e = Arc::clone(&e);
            let b = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                b.wait();
                for i in 0..PER_WRITER {
                    let k = format!("key_{w}_{i:03}");
                    assert!(e.insert(&k, (w * PER_WRITER + i) as u32));
                    assert!(e.contains(&k));
                    if i % 4 == 0 {
                        let r = format!("moved_{w}_{i:03}");
                        assert!(e.rename(&k, &r));
                        assert!(e.delete(&r));
                    }
                }
            }));
        }
        for _ in 0..READERS {
            let e = Arc::clone(&e);
            let b = Arc::clone(&barrier);
            let reads = Arc::clone(&reads);
            handles.push(std::thread::spawn(move || {
                b.wait();
                for _ in 0..200 {
                    // Prefiltered + full-scan queries racing the writers.
                    let r = e.search("key_.*").unwrap();
                    for k in r.keys() {
                        assert!(k.starts_with("key_"));
                    }
                    reads.fetch_add(e.search("key_0_0.*").unwrap().len(), Ordering::Relaxed);
                    let _ = e.len();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        // Surviving keys: everything except the i % 4 == 0 ones (deleted).
        let expected = WRITERS * (PER_WRITER - PER_WRITER.div_ceil(4));
        assert_eq!(e.len(), expected);
        assert_eq!(e.search("key_.*").unwrap().len(), expected);
        assert!(e.search("moved_.*").unwrap().is_empty());
        // No torn state: every surviving key is findable.
        for k in e.search("key_.*").unwrap().keys() {
            assert!(e.contains(k));
        }
    }

    #[test]
    fn search_snapshot_is_stable_under_concurrent_clear() {
        let e = Arc::new(seeded());
        let r = e.search(".*").unwrap();
        let before = r.keys().to_vec();
        let e2 = Arc::clone(&e);
        std::thread::spawn(move || e2.clear()).join().unwrap();
        assert_eq!(r.keys(), before.as_slice()); // snapshot untouched
        assert!(r.iter().next().is_none()); // values all gone -> skipped
        assert!(e.is_empty());
    }
}
