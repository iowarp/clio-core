// SPDX-License-Identifier: BSD-3-Clause
// Part of IOWarp Core — CTP Rust adaptation (issue #756).

//! Port of `clio_ctp/solver/nonlinear_least_squares.h` (`ctp::NonlinearLeastSquares`).
//!
//! A Levenberg–Marquardt nonlinear least-squares minimizer with a
//! forward-difference Jacobian and a dense Gauss–Jordan matrix inverse. The
//! arithmetic is a statement-for-statement transcription of the C++: same
//! operation order, same constants, same accept/reject rule, same quirks.
//!
//! C++ → Rust parity map
//! ---------------------
//! | C++ name                                        | Rust name                          |
//! |-------------------------------------------------|------------------------------------|
//! | `ctp::NonlinearLeastSquares`                    | [`NonlinearLeastSquares`]          |
//! | `NonlinearLeastSquares() = default`             | [`NonlinearLeastSquares::new`] / `Default` |
//! | `SetParameters(const std::vector<double>&)`     | [`NonlinearLeastSquares::set_parameters`] |
//! | `SetTolerance(double)`                          | [`NonlinearLeastSquares::set_tolerance`] |
//! | `SetMaxIterations(int)`                         | [`NonlinearLeastSquares::set_max_iterations`] |
//! | `SetLambda(double)`                             | [`NonlinearLeastSquares::set_lambda`] |
//! | `GetParameters() -> const vector<double>&`      | [`NonlinearLeastSquares::parameters`] `-> &[f64]` |
//! | `GetSumOfSquares() -> double`                   | [`NonlinearLeastSquares::sum_of_squares`] |
//! | `GetResiduals() -> const vector<double>&`       | [`NonlinearLeastSquares::residuals`] `-> &[f64]` |
//! | `Minimize(cost_func, args...) -> bool`          | [`NonlinearLeastSquares::minimize`] `-> bool` |
//! | `ComputeJacobian(cost_func, args...)` (private) | `compute_jacobian` (private)       |
//! | `ComputeSumOfSquares()` (private)               | `compute_sum_of_squares` (private) |
//! | `MatrixMultiply(A, B)` (private)                | `matrix_multiply` (private fn)     |
//! | `MatrixTranspose(M)` (private)                  | `matrix_transpose` (private fn)    |
//! | `MatrixInverse(M)` (private)                    | `matrix_inverse` (private fn)      |
//! | `MatrixVectorMultiply(M, v)` (private)          | `matrix_vector_multiply` (private fn) |
//! | `lambda_ = 0.001`                               | `lambda`, [`DEFAULT_LAMBDA`]       |
//! | `lambda_factor_ = 10.0`                         | `lambda_factor`, [`LAMBDA_FACTOR`] (no setter, as in C++) |
//! | `tolerance_ = 1e-8`                             | `tolerance`, [`DEFAULT_TOLERANCE`] |
//! | `max_iterations_ = 100`                         | `max_iterations`, [`DEFAULT_MAX_ITERATIONS`] |
//! | `const double h = 1e-8` (ComputeJacobian)       | `JACOBIAN_STEP`                    |
//! | `1e-12` pivot floor (MatrixInverse)             | `PIVOT_EPSILON`                    |
//! | `1e12` damping cap (Minimize)                   | `LAMBDA_MAX`                       |
//!
//! Faithfully preserved C++ quirks (NOT bugs introduced here)
//! ---------------------------------------------------------
//! * **`minimize` almost always returns `true`.** The invariant
//!   `prev_sum_of_squares == sum_of_squares` holds at the end of every
//!   iteration (both are updated together on an accepted step, and neither is
//!   touched on a rejected one), so the trailing
//!   `|prev - sos| < tolerance` reduces to `0.0 < tolerance`. A `true` return
//!   therefore means "did not diverge", not "converged": it is returned even
//!   when every step was rejected, when `max_iterations` is 0, and when the
//!   damping cap breaks the loop. It is `false` only for empty parameters, a
//!   non-positive `tolerance`, or NaN sums. Callers wanting a real convergence
//!   test should inspect [`sum_of_squares`](NonlinearLeastSquares::sum_of_squares).
//! * **`lambda` is persistent solver state**, never reset by `minimize`. A
//!   second `minimize` call starts from wherever the first left the damping
//!   (possibly `> 1e12`, which makes the loop break after one iteration).
//! * **The early-exit convergence test compares against the pre-step
//!   `prev`** and only runs on accepted steps; rejected steps cannot converge.
//! * **The Jacobian divides by the literal step `h`**, not by the realized
//!   `(p + h) - p`, so it inherits the C++ round-off exactly.
//! * **The initial `cost_func` call receives the previous residual buffer**
//!   (C++ passes the `residuals_` member, which is not cleared first); only
//!   later calls get a fresh empty buffer. Cost functions that append rather
//!   than assign will see leftovers on a second `minimize`.
//! * **`max_iterations` is `int`/`i32`**; zero or negative skips the loop.
//!
//! Semantic divergences (explicit)
//! -------------------------------
//! 1. **Variadic `Args&&...` dropped.** The C++ threads extra arguments
//!    through `Minimize`/`ComputeJacobian` into `cost_func`. Rust closures
//!    capture their environment, so [`minimize`](NonlinearLeastSquares::minimize)
//!    takes only the cost function; captured state replaces the pack. The
//!    C++ perfect-forwards the *same* pack to every call, so rvalue arguments
//!    that a cost function moves from would already be use-after-move there —
//!    capture-by-reference has no such hazard.
//! 2. **`FnMut`, not `const CostFunction&`.** C++ takes the functor by const
//!    reference (requiring a `const operator()`, but permitting `mutable`
//!    members to be mutated). `FnMut` is the strict superset that accepts
//!    every `Fn` closure and also plain mutable-capture closures.
//! 3. **Zero residuals is UB in C++, a `false` return here.** If `cost_func`
//!    produces an empty residual vector, `MatrixTranspose(jacobian_)` reads
//!    `matrix[0]` on an empty vector — UB. This port returns `false` from
//!    `minimize` immediately after the initial evaluation instead. (Reached
//!    only via a degenerate cost function; a real one always emits residuals.)
//! 4. **Malformed dimensions panic instead of invoking UB.** A `cost_func`
//!    that returns *fewer* residuals than the previous call makes C++ read out
//!    of bounds; Rust panics on the bounds check. Growing the residual count
//!    between iterations is fine in both (the C++ picks up the new length on
//!    the next iteration, and so does this port). The private matrix helpers
//!    treat an empty operand as having zero columns (`A[0]` is UB in C++)
//!    rather than reading past the end.
//! 5. **`std::swap(aug[i], aug[pivot_row])` → `Vec::swap`** — a row-handle
//!    swap in both, no element copies, numerically identical.
//! 6. No `unsafe` anywhere in this module; no shared memory, no threading, and
//!    no `thread_local` (per project rule). `NonlinearLeastSquares` is a plain
//!    owned value: `Send + Sync`, cloneable, with no `Drop` side effects.

/// Initial Levenberg–Marquardt damping (C++ `lambda_ = 0.001`).
pub const DEFAULT_LAMBDA: f64 = 0.001;

/// Damping step factor (C++ `lambda_factor_ = 10.0`). No C++ setter exists.
pub const LAMBDA_FACTOR: f64 = 10.0;

/// Convergence tolerance on the change in the sum of squares
/// (C++ `tolerance_ = 1e-8`).
pub const DEFAULT_TOLERANCE: f64 = 1e-8;

/// Iteration cap (C++ `max_iterations_ = 100`).
pub const DEFAULT_MAX_ITERATIONS: i32 = 100;

/// Forward-difference step for the numeric Jacobian
/// (C++ `const double h = 1e-8`).
const JACOBIAN_STEP: f64 = 1e-8;

/// Pivot magnitude below which `matrix_inverse` regularizes the diagonal
/// (C++ `1e-12`).
const PIVOT_EPSILON: f64 = 1e-12;

/// Damping value above which `minimize` abandons the loop (C++ `1e12`).
const LAMBDA_MAX: f64 = 1e12;

/// Levenberg–Marquardt nonlinear least-squares solver.
///
/// Port of `ctp::NonlinearLeastSquares`. The cost function receives the
/// current parameters and fills a residual vector; the solver drives the sum
/// of squared residuals downward.
///
/// ```
/// use ctp_solver::solver::NonlinearLeastSquares;
///
/// // Fit the line y = a*x + b to points that lie exactly on y = 2x + 1.
/// let data = [(0.0, 1.0), (1.0, 3.0), (2.0, 5.0)];
/// let mut solver = NonlinearLeastSquares::new();
/// solver.set_parameters(&[0.0, 0.0]);
/// solver.minimize(|p: &[f64], out: &mut Vec<f64>| {
///     out.clear();
///     for (x, y) in data {
///         out.push(p[0] * x + p[1] - y);
///     }
/// });
/// assert!((solver.parameters()[0] - 2.0).abs() < 1e-3);
/// assert!((solver.parameters()[1] - 1.0).abs() < 1e-3);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct NonlinearLeastSquares {
    /// C++ `sum_of_squares_`.
    sum_of_squares: f64,
    /// C++ `parameters_`.
    parameters: Vec<f64>,
    /// C++ `residuals_`.
    residuals: Vec<f64>,
    /// C++ `jacobian_`, row-major `[n_residuals][n_params]`.
    jacobian: Vec<Vec<f64>>,
    /// C++ `lambda_` — persistent across `minimize` calls, as in C++.
    lambda: f64,
    /// C++ `lambda_factor_` — no setter in C++, none here.
    lambda_factor: f64,
    /// C++ `tolerance_`.
    tolerance: f64,
    /// C++ `max_iterations_`.
    max_iterations: i32,
}

impl Default for NonlinearLeastSquares {
    /// Mirrors the C++ default member initializers (`= default` ctor).
    fn default() -> Self {
        Self {
            sum_of_squares: 0.0,
            parameters: Vec::new(),
            residuals: Vec::new(),
            jacobian: Vec::new(),
            lambda: DEFAULT_LAMBDA,
            lambda_factor: LAMBDA_FACTOR,
            tolerance: DEFAULT_TOLERANCE,
            max_iterations: DEFAULT_MAX_ITERATIONS,
        }
    }
}

impl NonlinearLeastSquares {
    /// C++ `NonlinearLeastSquares() = default`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// C++ `SetParameters(const std::vector<double>& initial_params)`.
    pub fn set_parameters(&mut self, initial_params: &[f64]) {
        self.parameters = initial_params.to_vec();
    }

    /// C++ `SetTolerance(double tol)`.
    pub fn set_tolerance(&mut self, tol: f64) {
        self.tolerance = tol;
    }

    /// C++ `SetMaxIterations(int max_iter)`.
    pub fn set_max_iterations(&mut self, max_iter: i32) {
        self.max_iterations = max_iter;
    }

    /// C++ `SetLambda(double lambda)`.
    pub fn set_lambda(&mut self, lambda: f64) {
        self.lambda = lambda;
    }

    /// C++ `GetParameters()`.
    #[must_use]
    pub fn parameters(&self) -> &[f64] {
        &self.parameters
    }

    /// C++ `GetSumOfSquares()`.
    #[must_use]
    pub fn sum_of_squares(&self) -> f64 {
        self.sum_of_squares
    }

    /// C++ `GetResiduals()`.
    #[must_use]
    pub fn residuals(&self) -> &[f64] {
        &self.residuals
    }

    /// C++ `ComputeSumOfSquares()` — sequential accumulation, same order.
    fn compute_sum_of_squares(&self) -> f64 {
        let mut sum = 0.0;
        for &r in &self.residuals {
            sum += r * r;
        }
        sum
    }

    /// C++ `ComputeJacobian` — forward differences with the literal step `h`.
    ///
    /// # Panics
    /// If `cost_func` yields fewer residuals than the current residual count
    /// (an out-of-bounds *read* in C++; see divergence 4).
    fn compute_jacobian<F>(&mut self, cost_func: &mut F)
    where
        F: FnMut(&[f64], &mut Vec<f64>),
    {
        let n_params = self.parameters.len();
        let n_residuals = self.residuals.len();

        self.jacobian = vec![vec![0.0; n_params]; n_residuals];

        for j in 0..n_params {
            // Forward difference.
            let mut params_plus = self.parameters.clone();
            params_plus[j] += JACOBIAN_STEP;

            let mut residuals_plus = Vec::new();
            cost_func(&params_plus, &mut residuals_plus);

            // Index loop retained deliberately: the bounds check on
            // `residuals_plus` is this port's defined-behavior stand-in for the
            // C++ out-of-bounds read (divergence 4). A `zip`/`take` form would
            // silently leave zero columns instead of reporting the mismatch.
            #[allow(clippy::needless_range_loop)]
            for i in 0..n_residuals {
                // C++ divides by the literal `h`, not by the realized step.
                self.jacobian[i][j] = (residuals_plus[i] - self.residuals[i]) / JACOBIAN_STEP;
            }
        }
    }

    /// C++ `Minimize(const CostFunction& cost_func, Args&&... args)`.
    ///
    /// `cost_func(params, residuals_out)` fills `residuals_out` for `params`.
    /// Returns the C++ `bool`, whose meaning is weaker than it looks — see the
    /// module docs: it is essentially "did not diverge", not "converged".
    ///
    /// # Panics
    /// If `cost_func` shrinks the residual count between calls (divergence 4).
    pub fn minimize<F>(&mut self, mut cost_func: F) -> bool
    where
        F: FnMut(&[f64], &mut Vec<f64>),
    {
        if self.parameters.is_empty() {
            return false;
        }

        // Initial evaluation. As in C++, the *existing* residual buffer is
        // handed to the cost function (it is not cleared first).
        cost_func(&self.parameters, &mut self.residuals);
        self.sum_of_squares = self.compute_sum_of_squares();

        // Divergence 3: C++ would dereference `matrix[0]` on an empty
        // Jacobian (UB). Report failure instead.
        if self.residuals.is_empty() {
            return false;
        }

        let mut prev_sum_of_squares = self.sum_of_squares;

        for _iter in 0..self.max_iterations {
            // Compute Jacobian.
            self.compute_jacobian(&mut cost_func);

            // Compute J^T * J and J^T * r.
            let j_t = matrix_transpose(&self.jacobian);
            let mut jtj = matrix_multiply(&j_t, &self.jacobian);
            let jtr = matrix_vector_multiply(&j_t, &self.residuals);

            // Add damping term (Levenberg-Marquardt). `jtj` is always
            // n_params x n_params here (J^T is n_params x n_residuals and the
            // Jacobian rows are n_params wide), so `take` never truncates.
            let n_params = self.parameters.len();
            for (i, row) in jtj.iter_mut().enumerate().take(n_params) {
                row[i] += self.lambda;
            }

            // Solve (J^T * J + lambda*I) * delta = -J^T * r.
            let jtj_inv = matrix_inverse(&jtj);
            let mut delta = matrix_vector_multiply(&jtj_inv, &jtr);

            // Negate delta (we want to minimize).
            for d in &mut delta {
                *d = -*d;
            }

            // Try new parameters.
            let mut new_params = self.parameters.clone();
            for i in 0..n_params {
                new_params[i] += delta[i];
            }

            // Evaluate cost at new parameters.
            let mut new_residuals = Vec::new();
            cost_func(&new_params, &mut new_residuals);

            let mut new_sum_of_squares = 0.0;
            for &r in &new_residuals {
                new_sum_of_squares += r * r;
            }

            // Check if improvement.
            if new_sum_of_squares < self.sum_of_squares {
                // Accept step.
                self.parameters = new_params;
                self.residuals = new_residuals;
                self.sum_of_squares = new_sum_of_squares;
                self.lambda /= self.lambda_factor; // Decrease damping.

                // Check convergence (against the pre-step `prev`).
                if (prev_sum_of_squares - self.sum_of_squares).abs() < self.tolerance {
                    return true;
                }
                prev_sum_of_squares = self.sum_of_squares;
            } else {
                // Reject step, increase damping.
                self.lambda *= self.lambda_factor;
            }

            // Prevent lambda from becoming too large.
            if self.lambda > LAMBDA_MAX {
                break;
            }
        }

        (prev_sum_of_squares - self.sum_of_squares).abs() < self.tolerance
    }
}

/// C++ `MatrixMultiply(A, B)` — naive triple loop, same accumulation order.
///
/// An empty operand is treated as having zero columns (`A[0]` is UB in C++).
///
/// # Panics
/// If `B` has fewer rows than `A` has columns, or its rows are ragged
/// (an out-of-bounds read in C++).
fn matrix_multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows_a = a.len();
    let cols_a = a.first().map_or(0, Vec::len);
    let cols_b = b.first().map_or(0, Vec::len);

    let mut result = vec![vec![0.0; cols_b]; rows_a];

    for (i, row) in result.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let mut acc = 0.0;
            for k in 0..cols_a {
                acc += a[i][k] * b[k][j];
            }
            *cell = acc;
        }
    }
    result
}

/// C++ `MatrixTranspose(matrix)`.
///
/// An empty input yields an empty output (`matrix[0]` is UB in C++).
fn matrix_transpose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = matrix.len();
    let cols = matrix.first().map_or(0, Vec::len);

    let mut result = vec![vec![0.0; rows]; cols];

    for (i, row) in matrix.iter().enumerate() {
        for (j, &value) in row.iter().enumerate().take(cols) {
            result[j][i] = value;
        }
    }
    result
}

/// C++ `MatrixInverse(matrix)` — Gauss-Jordan on `[A|I]` with partial
/// pivoting and a `1e-12` diagonal regularization for (near-)singular pivots.
///
/// Note the C++ regularizes by *adding* `1e-12` to the pivot rather than
/// bailing out, so this never fails; a singular input yields the same huge
/// finite entries here as there.
fn matrix_inverse(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = matrix.len();
    let mut aug = vec![vec![0.0; 2 * n]; n];

    // Create augmented matrix [A|I].
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = matrix[i][j];
        }
        aug[i][i + n] = 1.0;
    }

    // Gauss-Jordan elimination.
    for i in 0..n {
        // Find pivot.
        let mut pivot_row = i;
        for k in (i + 1)..n {
            if aug[k][i].abs() > aug[pivot_row][i].abs() {
                pivot_row = k;
            }
        }

        // Swap rows if needed.
        if pivot_row != i {
            aug.swap(i, pivot_row);
        }

        // Make diagonal element 1.
        let mut pivot = aug[i][i];
        if pivot.abs() < PIVOT_EPSILON {
            // Singular matrix, add small regularization.
            aug[i][i] += PIVOT_EPSILON;
            pivot = aug[i][i];
        }

        for value in &mut aug[i] {
            *value /= pivot;
        }

        // Eliminate column. The pivot row is untouched by this loop (k != i),
        // so cloning it once is numerically identical to C++'s aliased reads.
        let pivot_row_values = aug[i].clone();
        for (k, row) in aug.iter_mut().enumerate() {
            if k != i {
                let factor = row[i];
                for (value, &pivot_value) in row.iter_mut().zip(pivot_row_values.iter()) {
                    *value -= factor * pivot_value;
                }
            }
        }
    }

    // Extract inverse matrix.
    let mut inverse = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inverse[i][j] = aug[i][j + n];
        }
    }
    inverse
}

/// C++ `MatrixVectorMultiply(matrix, vector)`.
///
/// # Panics
/// If `vector` is shorter than the matrix's column count (an out-of-bounds
/// read in C++).
fn matrix_vector_multiply(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    let rows = matrix.len();
    let cols = matrix.first().map_or(0, Vec::len);

    let mut result = vec![0.0; rows];

    for (i, cell) in result.iter_mut().enumerate() {
        for j in 0..cols {
            *cell += matrix[i][j] * vector[j];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute closeness helper for hand-computed expectations.
    fn assert_close(actual: f64, expected: f64, eps: f64) {
        assert!(
            (actual - expected).abs() <= eps,
            "expected {expected}, got {actual} (eps {eps})"
        );
    }

    /// Relative closeness, for values whose magnitude makes absolute eps moot.
    fn assert_rel_close(actual: f64, expected: f64, rel: f64) {
        let tol = expected.abs() * rel;
        assert!(
            (actual - expected).abs() <= tol,
            "expected {expected}, got {actual} (rel {rel})"
        );
    }

    // ---------------------------------------------------------------
    // Defaults / accessors
    // ---------------------------------------------------------------

    #[test]
    fn defaults_match_cpp_member_initializers() {
        let s = NonlinearLeastSquares::new();
        assert_eq!(s.sum_of_squares(), 0.0);
        assert!(s.parameters().is_empty());
        assert!(s.residuals().is_empty());
        assert!(s.jacobian.is_empty());
        assert_eq!(s.lambda, 0.001);
        assert_eq!(s.lambda_factor, 10.0);
        assert_eq!(s.tolerance, 1e-8);
        assert_eq!(s.max_iterations, 100);
        assert_eq!(s, NonlinearLeastSquares::default());
    }

    #[test]
    fn setters_round_trip() {
        let mut s = NonlinearLeastSquares::new();
        s.set_parameters(&[1.0, 2.0, 3.0]);
        s.set_tolerance(1e-4);
        s.set_max_iterations(7);
        s.set_lambda(0.5);
        assert_eq!(s.parameters(), &[1.0, 2.0, 3.0]);
        assert_eq!(s.tolerance, 1e-4);
        assert_eq!(s.max_iterations, 7);
        assert_eq!(s.lambda, 0.5);

        // SetParameters copies (C++ takes a const ref and assigns).
        s.set_parameters(&[9.0]);
        assert_eq!(s.parameters(), &[9.0]);
    }

    /// No shared/thread-affine state: the solver is a plain owned value.
    #[test]
    fn solver_is_send_and_sync() {
        fn require<T: Send + Sync>() {}
        require::<NonlinearLeastSquares>();
    }

    // ---------------------------------------------------------------
    // matrix_multiply
    // ---------------------------------------------------------------

    #[test]
    fn matrix_multiply_hand_computed() {
        // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        assert_eq!(
            matrix_multiply(&a, &b),
            vec![vec![19.0, 22.0], vec![43.0, 50.0]]
        );
    }

    #[test]
    fn matrix_multiply_non_square() {
        // (2x3) * (3x1): [[1,2,3],[4,5,6]] * [[1],[0],[-1]] = [[-2],[-2]]
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let b = vec![vec![1.0], vec![0.0], vec![-1.0]];
        assert_eq!(matrix_multiply(&a, &b), vec![vec![-2.0], vec![-2.0]]);
    }

    #[test]
    fn matrix_multiply_empty_operands_yield_empty() {
        // C++ reads A[0] / B[0] here (UB); this port defines them as 0 columns.
        assert!(matrix_multiply(&[], &[]).is_empty());
        assert!(matrix_multiply(&[], &[vec![1.0]]).is_empty());
        // A non-empty with B empty: zero columns out, rows preserved.
        assert_eq!(
            matrix_multiply(&[vec![1.0, 2.0]], &[]),
            vec![Vec::<f64>::new()]
        );
        // A row of width zero: J^T * J when there are no parameters.
        assert_eq!(matrix_multiply(&[vec![]], &[vec![]]), vec![Vec::<f64>::new()]);
    }

    // ---------------------------------------------------------------
    // matrix_transpose
    // ---------------------------------------------------------------

    #[test]
    fn matrix_transpose_hand_computed() {
        let m = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        assert_eq!(
            matrix_transpose(&m),
            vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]
        );
    }

    #[test]
    fn matrix_transpose_involution() {
        let m = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        assert_eq!(matrix_transpose(&matrix_transpose(&m)), m);
    }

    #[test]
    fn matrix_transpose_empty_and_zero_width() {
        assert!(matrix_transpose(&[]).is_empty());
        // 2 rows of width 0 -> 0 rows (C++ would read matrix[0][...] fine but
        // produce the same empty result).
        assert!(matrix_transpose(&[vec![], vec![]]).is_empty());
    }

    // ---------------------------------------------------------------
    // matrix_inverse
    // ---------------------------------------------------------------

    #[test]
    fn matrix_inverse_hand_computed_2x2() {
        // [[4,7],[2,6]], det = 10 -> [[0.6,-0.7],[-0.2,0.4]]
        let m = vec![vec![4.0, 7.0], vec![2.0, 6.0]];
        let inv = matrix_inverse(&m);
        assert_close(inv[0][0], 0.6, 1e-12);
        assert_close(inv[0][1], -0.7, 1e-12);
        assert_close(inv[1][0], -0.2, 1e-12);
        assert_close(inv[1][1], 0.4, 1e-12);
    }

    #[test]
    fn matrix_inverse_identity_is_exact() {
        let m = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert_eq!(matrix_inverse(&m), m);
    }

    #[test]
    fn matrix_inverse_partial_pivot_swap() {
        // [[0,1],[1,0]] is its own inverse and forces the pivot swap branch
        // (|aug[1][0]| = 1 > |aug[0][0]| = 0).
        let m = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        assert_eq!(matrix_inverse(&m), vec![vec![0.0, 1.0], vec![1.0, 0.0]]);
    }

    #[test]
    fn matrix_inverse_diagonal_scaling() {
        let m = vec![vec![2.0, 0.0, 0.0], vec![0.0, 4.0, 0.0], vec![0.0, 0.0, 8.0]];
        let inv = matrix_inverse(&m);
        assert_eq!(
            inv,
            vec![
                vec![0.5, 0.0, 0.0],
                vec![0.0, 0.25, 0.0],
                vec![0.0, 0.0, 0.125]
            ]
        );
    }

    #[test]
    fn matrix_inverse_round_trip_3x3() {
        let m = vec![
            vec![2.0, -1.0, 0.0],
            vec![-1.0, 2.0, -1.0],
            vec![0.0, -1.0, 2.0],
        ];
        let inv = matrix_inverse(&m);
        let prod = matrix_multiply(&m, &inv);
        for (i, row) in prod.iter().enumerate() {
            for (j, &cell) in row.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_close(cell, expected, 1e-12);
            }
        }
    }

    #[test]
    fn matrix_inverse_singular_uses_pivot_regularization() {
        // [[1,1],[1,1]] is singular. Hand-tracing the C++: the second pivot is
        // 0 -> += 1e-12, giving inverse ~= [[1e12+1, -1e12], [-1e12, 1e12]].
        // No panic, no NaN — huge finite entries, exactly as in C++.
        let m = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        let inv = matrix_inverse(&m);
        assert_rel_close(inv[0][0], 1e12, 1e-6);
        assert_rel_close(inv[0][1], -1e12, 1e-6);
        assert_rel_close(inv[1][0], -1e12, 1e-6);
        assert_rel_close(inv[1][1], 1e12, 1e-6);
        assert!(inv.iter().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn matrix_inverse_zero_matrix_is_regularized_not_nan() {
        // Every pivot is 0 -> every pivot becomes 1e-12; 1/1e-12 = 1e12.
        let m = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let inv = matrix_inverse(&m);
        assert_rel_close(inv[0][0], 1e12, 1e-6);
        assert_rel_close(inv[1][1], 1e12, 1e-6);
        assert_close(inv[0][1], 0.0, 1e-12);
        assert_close(inv[1][0], 0.0, 1e-12);
    }

    #[test]
    fn matrix_inverse_empty_is_empty() {
        assert!(matrix_inverse(&[]).is_empty());
    }

    // ---------------------------------------------------------------
    // matrix_vector_multiply
    // ---------------------------------------------------------------

    #[test]
    fn matrix_vector_multiply_hand_computed() {
        // [[1,2],[3,4]] * [5,6] = [17, 39]
        let m = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert_eq!(matrix_vector_multiply(&m, &[5.0, 6.0]), vec![17.0, 39.0]);
    }

    #[test]
    fn matrix_vector_multiply_empty_and_extra_elements() {
        assert!(matrix_vector_multiply(&[], &[1.0]).is_empty());
        // Extra vector elements past the column count are ignored (as in C++).
        let m = vec![vec![1.0, 1.0]];
        assert_eq!(matrix_vector_multiply(&m, &[2.0, 3.0, 99.0]), vec![5.0]);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn matrix_vector_multiply_short_vector_panics_instead_of_ub() {
        // Divergence 4: C++ reads out of bounds here.
        let m = vec![vec![1.0, 2.0]];
        let _ = matrix_vector_multiply(&m, &[1.0]);
    }

    // ---------------------------------------------------------------
    // compute_sum_of_squares / compute_jacobian
    // ---------------------------------------------------------------

    #[test]
    fn sum_of_squares_empty_is_zero() {
        let s = NonlinearLeastSquares::new();
        assert_eq!(s.compute_sum_of_squares(), 0.0);
    }

    #[test]
    fn sum_of_squares_hand_computed() {
        let mut s = NonlinearLeastSquares::new();
        s.residuals = vec![1.0, -2.0, 3.0];
        assert_eq!(s.compute_sum_of_squares(), 14.0); // 1 + 4 + 9
    }

    #[test]
    fn jacobian_of_linear_model_is_the_design_matrix() {
        // r_i(a, b) = a*x_i + b - y_i  =>  dr/da = x_i, dr/db = 1.
        let xs = [0.0, 1.0, 2.0];
        let mut cost = |p: &[f64], out: &mut Vec<f64>| {
            out.clear();
            for &x in &xs {
                out.push(p[0] * x + p[1]);
            }
        };

        let mut s = NonlinearLeastSquares::new();
        s.set_parameters(&[1.0, 0.0]);
        cost(&s.parameters.clone(), &mut s.residuals);
        s.compute_jacobian(&mut cost);

        assert_eq!(s.jacobian.len(), 3);
        for (i, &x) in xs.iter().enumerate() {
            assert_eq!(s.jacobian[i].len(), 2);
            assert_close(s.jacobian[i][0], x, 1e-6);
            assert_close(s.jacobian[i][1], 1.0, 1e-6);
        }
    }

    #[test]
    fn jacobian_of_quadratic_matches_analytic_derivative() {
        // r(p) = p^2 => dr/dp = 2p; forward difference at p=3 gives 6 + h.
        let mut cost = |p: &[f64], out: &mut Vec<f64>| {
            out.clear();
            out.push(p[0] * p[0]);
        };
        let mut s = NonlinearLeastSquares::new();
        s.set_parameters(&[3.0]);
        cost(&s.parameters.clone(), &mut s.residuals);
        s.compute_jacobian(&mut cost);
        assert_close(s.jacobian[0][0], 6.0, 1e-5);
    }

    #[test]
    fn jacobian_shape_with_zero_params_or_zero_residuals() {
        let mut cost = |_p: &[f64], out: &mut Vec<f64>| out.clear();

        // No parameters: no columns to difference, cost_func never called.
        let mut s = NonlinearLeastSquares::new();
        s.residuals = vec![1.0, 2.0];
        s.compute_jacobian(&mut cost);
        assert_eq!(s.jacobian, vec![Vec::<f64>::new(), Vec::<f64>::new()]);

        // No residuals: zero rows.
        let mut s2 = NonlinearLeastSquares::new();
        s2.set_parameters(&[1.0]);
        s2.compute_jacobian(&mut cost);
        assert!(s2.jacobian.is_empty());
    }

    // ---------------------------------------------------------------
    // minimize — happy paths
    // ---------------------------------------------------------------

    #[test]
    fn minimize_single_parameter_root() {
        // r(p) = p - 3 => minimum at p = 3.
        let mut s = NonlinearLeastSquares::new();
        s.set_parameters(&[0.0]);
        let ok = s.minimize(|p: &[f64], out: &mut Vec<f64>| {
            out.clear();
            out.push(p[0] - 3.0);
        });
        assert!(ok);
        assert_close(s.parameters()[0], 3.0, 1e-6);
        assert!(s.sum_of_squares() < 1e-12);
        assert_eq!(s.residuals().len(), 1);
    }

    #[test]
    fn minimize_linear_fit_exact_data() {
        // Points exactly on y = 2x + 1.
        let data = [(0.0, 1.0), (1.0, 3.0), (2.0, 5.0)];
        let mut s = NonlinearLeastSquares::new();
        s.set_parameters(&[0.0, 0.0]);
        let ok = s.minimize(|p: &[f64], out: &mut Vec<f64>| {
            out.clear();
            for (x, y) in data {
                out.push(p[0] * x + p[1] - y);
            }
        });
        assert!(ok);
        assert_close(s.parameters()[0], 2.0, 1e-3);
        assert_close(s.parameters()[1], 1.0, 1e-3);
        assert!(s.sum_of_squares() < 1e-6);
    }

    #[test]
    fn minimize_overdetermined_fit_reaches_normal_equation_solution() {
        // y = a*x + b through (0,0), (1,1), (2,2), (3,10). The data does not
        // lie on a line, so the optimum is the ordinary-least-squares
        // solution of the normal equations. Hand-computed:
        //   n = 4, Sx = 6, Sy = 13, Sxx = 14, Sxy = 0+1+4+30 = 35
        //   a = (n*Sxy - Sx*Sy) / (n*Sxx - Sx^2) = (140 - 78)/(56 - 36) = 3.1
        //   b = (Sy - a*Sx) / n = (13 - 18.6)/4 = -1.4
        // Residuals at the optimum: -1.4, 0.7, 2.8, -2.1
        //   => SoS = 1.96 + 0.49 + 7.84 + 4.41 = 14.7
        let data = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 10.0)];
        let mut s = NonlinearLeastSquares::new();
        s.set_parameters(&[0.0, 0.0]);
        s.set_tolerance(1e-14);
        s.set_max_iterations(500);
        s.minimize(|p: &[f64], out: &mut Vec<f64>| {
            out.clear();
            for (x, y) in data {
                out.push(p[0] * x + p[1] - y);
            }
        });
        assert_close(s.parameters()[0], 3.1, 1e-3);
        assert_close(s.parameters()[1], -1.4, 1e-3);
        assert_close(s.sum_of_squares(), 14.7, 1e-3);
        // The optimum is a stationary point of a nonzero-residual problem:
        // J^T * r must vanish there (both normal equations).
        let residuals = s.residuals();
        let sum_r: f64 = residuals.iter().sum();
        let sum_xr: f64 = residuals
            .iter()
            .zip(data.iter())
            .map(|(r, (x, _))| r * x)
            .sum();
        assert_close(sum_r, 0.0, 1e-3);
        assert_close(sum_xr, 0.0, 1e-3);
    }

    #[test]
    fn minimize_two_parameter_nonlinear_model() {
        // r_i = a*exp(b*x_i) - y_i with data generated from a=2, b=0.5.
        let xs: [f64; 5] = [0.0, 0.5, 1.0, 1.5, 2.0];
        let ys: Vec<f64> = xs.iter().map(|x| 2.0 * (0.5 * x).exp()).collect();
        let mut s = NonlinearLeastSquares::new();
        s.set_parameters(&[1.0, 1.0]);
        s.set_max_iterations(500);
        s.minimize(|p: &[f64], out: &mut Vec<f64>| {
            out.clear();
            for (i, &x) in xs.iter().enumerate() {
                out.push(p[0] * (p[1] * x).exp() - ys[i]);
            }
        });
        assert!(
            s.sum_of_squares() < 1e-6,
            "sum_of_squares = {}",
            s.sum_of_squares()
        );
        assert_close(s.parameters()[0], 2.0, 1e-2);
        assert_close(s.parameters()[1], 0.5, 1e-2);
    }

    #[test]
    fn minimize_already_at_optimum_keeps_parameters() {
        let mut s = NonlinearLeastSquares::new();
        s.set_parameters(&[3.0]);
        let ok = s.minimize(|p: &[f64], out: &mut Vec<f64>| {
            out.clear();
            out.push(p[0] - 3.0);
        });
        assert!(ok);
        assert_close(s.parameters()[0], 3.0, 1e-6);
        assert!(s.sum_of_squares() < 1e-12);
    }

    // ---------------------------------------------------------------
    // minimize — edge cases and preserved C++ quirks
    // ---------------------------------------------------------------

    #[test]
    fn minimize_empty_parameters_returns_false_without_calling_cost() {
        let mut calls = 0;
        let mut s = NonlinearLeastSquares::new();
        let ok = s.minimize(|_p: &[f64], out: &mut Vec<f64>| {
            calls += 1;
            out.push(1.0);
        });
        assert!(!ok);
        assert_eq!(calls, 0);
        assert_eq!(s.sum_of_squares(), 0.0);
    }

    #[test]
    fn minimize_zero_max_iterations_evaluates_once_and_returns_true() {
        // Quirk: prev == sos at loop exit, so |0| < tolerance -> true.
        let mut calls = 0;
        let mut s = NonlinearLeastSquares::new();
        s.set_parameters(&[5.0]);
        s.set_max_iterations(0);
        let ok = s.minimize(|p: &[f64], out: &mut Vec<f64>| {
            calls += 1;
            out.clear();
            out.push(p[0]);
        });
        assert!(ok, "C++ returns true here: |prev - sos| == 0 < tolerance");
        assert_eq!(calls, 1, "only the initial evaluation runs");
        assert_eq!(s.parameters(), &[5.0], "parameters untouched");
        assert_eq!(s.sum_of_squares(), 25.0, "initial evaluation still happens");
    }

    #[test]
    fn minimize_negative_max_iterations_behaves_like_zero() {
        let mut s = NonlinearLeastSquares::new();
        s.set_parameters(&[5.0]);
        s.set_max_iterations(-7);
        let ok = s.minimize(|p: &[f64], out: &mut Vec<f64>| {
            out.clear();
            out.push(p[0]);
        });
        assert!(ok);
        assert_eq!(s.parameters(), &[5.0]);
        assert_eq!(s.sum_of_squares(), 25.0);
    }

    #[test]
    fn minimize_non_positive_tolerance_returns_false() {
        // The only non-degenerate way to get `false`: |prev - sos| == 0 is
        // never < 0 (or < -1), so the early return and the final return both
        // fail. The solver still does its work.
        let mut s = NonlinearLeastSquares::new();
        s.set_parameters(&[0.0]);
        s.set_tolerance(0.0);
        s.set_max_iterations(10);
        let ok = s.minimize(|p: &[f64], out: &mut Vec<f64>| {
            out.clear();
            out.push(p[0] - 3.0);
        });
        assert!(!ok, "tolerance = 0 can never satisfy `< tolerance`");
        assert_close(s.parameters()[0], 3.0, 1e-6); // work still happened
    }

    #[test]
    fn minimize_returns_true_even_when_every_step_is_rejected() {
        // Quirk: a constant cost has a zero Jacobian, so delta = 0 and the new
        // sum of squares equals the old -> `new < old` is false every time.
        // Damping grows 0.001 * 10^k until it exceeds 1e12, the loop breaks,
        // and the final |prev - sos| == 0 < tolerance still returns true.
        let mut s = NonlinearLeastSquares::new();
        s.set_parameters(&[1.0]);
        let ok = s.minimize(|_p: &[f64], out: &mut Vec<f64>| {
            out.clear();
            out.push(1.0);
        });
        assert!(ok, "true means 'did not diverge', not 'converged'");
        assert_eq!(s.parameters(), &[1.0], "no step was ever accepted");
        assert_eq!(s.sum_of_squares(), 1.0, "cost never improved");
        assert!(s.lambda > LAMBDA_MAX, "damping hit the cap: {}", s.lambda);
    }

    #[test]
    fn lambda_persists_across_minimize_calls() {
        // Quirk: `minimize` never resets lambda_.
        let mut s = NonlinearLeastSquares::new();
        s.set_parameters(&[1.0]);
        let mut constant = |_p: &[f64], out: &mut Vec<f64>| {
            out.clear();
            out.push(1.0);
        };
        s.minimize(&mut constant);
        let after_first = s.lambda;
        assert!(after_first > LAMBDA_MAX);

        // Second call starts damped past the cap: it breaks after one
        // iteration (the check runs at the end of the body), leaving lambda
        // multiplied exactly once more.
        s.minimize(&mut constant);
        assert_eq!(s.lambda, after_first * LAMBDA_FACTOR);

        // set_lambda is the only way back.
        s.set_lambda(DEFAULT_LAMBDA);
        assert_eq!(s.lambda, DEFAULT_LAMBDA);
    }

    #[test]
    fn minimize_decreases_lambda_on_accepted_step() {
        let mut s = NonlinearLeastSquares::new();
        s.set_parameters(&[0.0]);
        s.set_max_iterations(1);
        s.minimize(|p: &[f64], out: &mut Vec<f64>| {
            out.clear();
            out.push(p[0] - 3.0);
        });
        // One accepted step: 0.001 / 10.
        assert_close(s.lambda, DEFAULT_LAMBDA / LAMBDA_FACTOR, 1e-18);
    }

    #[test]
    fn minimize_initial_call_receives_previous_residual_buffer() {
        // Quirk: C++ passes the `residuals_` member to the first cost_func
        // call without clearing it; only later calls get a fresh vector.
        let mut s = NonlinearLeastSquares::new();
        s.set_parameters(&[0.0]);
        s.minimize(|p: &[f64], out: &mut Vec<f64>| {
            out.clear();
            out.push(p[0] - 3.0);
        });
        assert_eq!(s.residuals().len(), 1);

        let mut first_call_len: Option<usize> = None;
        s.minimize(|p: &[f64], out: &mut Vec<f64>| {
            if first_call_len.is_none() {
                first_call_len = Some(out.len());
            }
            out.clear();
            out.push(p[0] - 3.0);
        });
        assert_eq!(
            first_call_len,
            Some(1),
            "the initial evaluation sees the leftover residual buffer"
        );
    }

    #[test]
    fn minimize_zero_residuals_returns_false_instead_of_ub() {
        // Divergence 3: C++ would index matrix[0] on an empty Jacobian.
        let mut s = NonlinearLeastSquares::new();
        s.set_parameters(&[1.0, 2.0]);
        let ok = s.minimize(|_p: &[f64], out: &mut Vec<f64>| out.clear());
        assert!(!ok);
        assert_eq!(s.sum_of_squares(), 0.0);
        assert!(s.residuals().is_empty());
        assert_eq!(s.parameters(), &[1.0, 2.0]);
    }

    #[test]
    fn minimize_nan_cost_terminates_and_returns_false() {
        // |NaN - NaN| < tol is false: no step is ever accepted, damping runs
        // to the cap, the loop breaks, and the final comparison is false.
        let mut s = NonlinearLeastSquares::new();
        s.set_parameters(&[1.0]);
        let ok = s.minimize(|_p: &[f64], out: &mut Vec<f64>| {
            out.clear();
            out.push(f64::NAN);
        });
        assert!(!ok);
        assert!(s.sum_of_squares().is_nan());
    }

    #[test]
    fn minimize_infinite_cost_does_not_hang() {
        let mut s = NonlinearLeastSquares::new();
        s.set_parameters(&[1.0]);
        let ok = s.minimize(|_p: &[f64], out: &mut Vec<f64>| {
            out.clear();
            out.push(f64::INFINITY);
        });
        // inf*inf = inf; `inf < inf` is false -> every step rejected -> the
        // damping cap breaks the loop -> |inf - inf| = NaN -> false.
        assert!(!ok);
        assert!(s.lambda > LAMBDA_MAX);
    }

    #[test]
    fn minimize_growing_residual_count_is_accepted() {
        // The residual count may grow between iterations (C++ picks up the new
        // length on the next pass); only shrinking is out-of-bounds.
        let mut n = 1usize;
        let mut s = NonlinearLeastSquares::new();
        s.set_parameters(&[0.0]);
        s.set_max_iterations(3);
        let ok = s.minimize(|p: &[f64], out: &mut Vec<f64>| {
            out.clear();
            for _ in 0..n {
                out.push((p[0] - 3.0) * 0.001);
            }
            n += 1;
        });
        assert!(ok);
        assert!(s.sum_of_squares().is_finite());
    }

    #[test]
    fn minimize_single_residual_many_params_is_rank_deficient_but_finite() {
        // 1 residual, 2 parameters: J^T*J is singular; the 1e-12 pivot
        // regularization plus lambda keep the solve finite (no panic, no NaN).
        let mut s = NonlinearLeastSquares::new();
        s.set_parameters(&[0.0, 0.0]);
        s.set_max_iterations(20);
        let ok = s.minimize(|p: &[f64], out: &mut Vec<f64>| {
            out.clear();
            out.push(p[0] + p[1] - 4.0);
        });
        assert!(ok);
        assert!(s.parameters().iter().all(|v| v.is_finite()));
        assert!(
            s.sum_of_squares() < 1.0,
            "should still descend: {}",
            s.sum_of_squares()
        );
    }

    #[test]
    fn minimize_accepts_a_plain_fn_pointer() {
        // C++ takes `const CostFunction&`; a stateless function works here too.
        fn cost(p: &[f64], out: &mut Vec<f64>) {
            out.clear();
            out.push(p[0] - 1.0);
        }
        let mut s = NonlinearLeastSquares::new();
        s.set_parameters(&[0.0]);
        assert!(s.minimize(cost as fn(&[f64], &mut Vec<f64>)));
        assert_close(s.parameters()[0], 1.0, 1e-6);
    }

    #[test]
    fn minimize_is_reentrant_on_a_fresh_solver() {
        // Two independent solvers do not interfere (no global/thread state).
        let mut a = NonlinearLeastSquares::new();
        let mut b = NonlinearLeastSquares::new();
        a.set_parameters(&[0.0]);
        b.set_parameters(&[0.0]);
        a.minimize(|p: &[f64], out: &mut Vec<f64>| {
            out.clear();
            out.push(p[0] - 3.0);
        });
        b.minimize(|p: &[f64], out: &mut Vec<f64>| {
            out.clear();
            out.push(p[0] + 7.0);
        });
        assert_close(a.parameters()[0], 3.0, 1e-6);
        assert_close(b.parameters()[0], -7.0, 1e-6);
    }
}
