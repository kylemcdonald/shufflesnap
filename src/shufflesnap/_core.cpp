// ShuffleSnap compiled kernels.
//
// Conventions shared with the Python layer:
//   P     (N, 2) float64   point coordinates, already normalized into grid units.
//   grid  (H, W) int32     cell id of every lattice position, or -1 if masked out.
//   C     (M, 2) float64   coordinates of every valid cell (x, y), same frame as P.
//   occ   (M,)   int32     point occupying each cell, or -1 when the cell is empty.
//   pos   (N,)   int32     cell of each point (inverse of occ).
// A lattice position is (x, y) with 0 <= x < W (column) and 0 <= y < H (row).

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "_dual.hpp"
#include "_lap.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace shufflesnap {

using Clock = std::chrono::steady_clock;

template <typename T>
using CArr1 = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
template <typename T>
using CArr2 = nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
template <typename T>
using MArr1 = nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

static int resolve_threads(int threads) {
#ifdef _OPENMP
    return threads > 0 ? threads : omp_get_max_threads();
#else
    (void)threads;
    return 1;
#endif
}

struct Problem {
    int64_t N = 0, M = 0;
    int W = 0, H = 0;
    const double* P = nullptr;
    const int32_t* grid = nullptr;
    const double* C = nullptr;
};

// A window is the set of lattice positions (a + i*sx, b + j*sy) for i in [i0, i1)
// and j in [j0, j1): a rectangle in the coordinates of residue subgrid (a, b).
struct Window {
    int32_t a, b, i0, i1, j0, j1;
};

// Tile every residue subgrid of stride (sx, sy) with w-by-w windows whose tiling is
// shifted by (ox, oy) subgrid cells.  Windows that fall partly outside the subgrid
// are clipped, never dropped, on all four sides, so every lattice position belongs to
// exactly one window of the phase.  Windows with fewer than two valid cells are
// omitted because nothing can move inside them.
static std::vector<Window> make_windows(const Problem& pb, int sx, int sy, int w, int ox, int oy) {
    std::vector<Window> out;
    const int shx = (w - (ox % w)) % w;
    const int shy = (w - (oy % w)) % w;
    for (int b = 0; b < sy && b < pb.H; ++b) {
        const int Hb = (pb.H - b + sy - 1) / sy;
        const int nwy = (Hb + shy + w - 1) / w;
        for (int a = 0; a < sx && a < pb.W; ++a) {
            const int Wa = (pb.W - a + sx - 1) / sx;
            const int nwx = (Wa + shx + w - 1) / w;
            for (int ky = 0; ky < nwy; ++ky) {
                const int v0 = ky * w - shy;
                const int j0 = std::max(0, v0), j1 = std::min(Hb, v0 + w);
                if (j0 >= j1) continue;
                for (int kx = 0; kx < nwx; ++kx) {
                    const int u0 = kx * w - shx;
                    const int i0 = std::max(0, u0), i1 = std::min(Wa, u0 + w);
                    if (i0 >= i1) continue;
                    int cnt = 0;
                    for (int j = j0; j < j1 && cnt < 2; ++j) {
                        const int64_t y = b + (int64_t)j * sy;
                        for (int i = i0; i < i1; ++i) {
                            if (pb.grid[y * pb.W + a + (int64_t)i * sx] >= 0 && ++cnt >= 2) break;
                        }
                    }
                    if (cnt >= 2) out.push_back({a, b, i0, i1, j0, j1});
                }
            }
        }
    }
    return out;
}

// Deterministic (thread-count independent) total squared displacement.
static double total_cost(const Problem& pb, const int32_t* pos, int nthreads) {
    const int64_t block = 8192;
    const int64_t nb_ = (pb.N + block - 1) / block;
    std::vector<double> part(std::max<int64_t>(nb_, 1), 0.0);
#pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int64_t bi = 0; bi < nb_; ++bi) {
        double s = 0.0;
        const int64_t e = std::min(pb.N, (bi + 1) * block);
        for (int64_t i = bi * block; i < e; ++i) {
            const int64_t c = pos[i];
            const double dx = pb.P[2 * i] - pb.C[2 * c];
            const double dy = pb.P[2 * i + 1] - pb.C[2 * c + 1];
            s += dx * dx + dy * dy;
        }
        part[bi] = s;
    }
    double s = 0.0;
    for (double v : part) s += v;
    return s;
}

struct PhaseStats {
    int64_t windows = 0, solved = 0, improved = 0, moved = 0;
    double gain = 0.0;
};

struct Deadline {
    bool active = false;
    Clock::time_point t;
    std::atomic<bool> hit{false};
    bool passed() {
        if (hit.load(std::memory_order_relaxed)) return true;
        if (active && Clock::now() >= t) {
            hit.store(true);
            return true;
        }
        return false;
    }
};

// Window LAP solvers by id (see _lap.hpp): 0 Hungarian, 1 Jonker-Volgenant,
// 2 Hungarian with greedy start, 3 auction with exact Hungarian finish, 4 geometric
// (quantile Brenier start; uses the coordinates P, C of the window's points and cells,
// row-major (x, y) pairs, when given).
struct WindowSolver {
    int id;
    LapWork lw;
    JvWork jw;
    AuctionWork aw;
    GeoWork gw;
    explicit WindowSolver(int solver) : id(solver) {}
    static void check(int solver) {
        if (solver < 0 || solver > 4) throw std::invalid_argument("unknown window solver");
    }
    double operator()(const double* a, int k, int m, int* row2col, const double* P = nullptr,
                      const double* C = nullptr) {
        switch (id) {
            case 1: return lap_jv(a, k, m, row2col, jw);
            case 2: return lap_hungarian_greedy(a, k, m, row2col, lw);
            case 3: return lap_auction(a, k, m, row2col, aw);
            case 4: return lap_geometric(a, k, m, row2col, gw, P, C);
            default: return lap_hungarian(a, k, m, row2col, lw);
        }
    }
};

// Solve every window of one phase exactly.  Windows are disjoint, so they are
// processed in parallel without locks.  A window whose cells have not changed
// occupant since it was last solved (stamp <= last_solved) is already optimal and
// is skipped when skip_clean is set; this never changes the result.
static PhaseStats run_phase(const Problem& pb, int32_t* occ, int32_t* pos, int64_t* stamp,
                            const std::vector<Window>& wins, std::vector<int64_t>& last_solved,
                            int sx, int sy, int w, int64_t phase_id, bool skip_clean, double tol_rel,
                            int nthreads, Deadline& dl, int solver) {
    PhaseStats st;
    st.windows = (int64_t)wins.size();
    const int maxc = w * w;
    const int64_t nw = (int64_t)wins.size();
    int64_t solved = 0, improved = 0, moved = 0;
    double gain = 0.0;
#pragma omp parallel num_threads(nthreads) reduction(+ : solved, improved, moved, gain)
    {
        std::vector<int32_t> cells(maxc), rows(maxc), curcol(maxc), oldocc(maxc), newocc(maxc);
        std::vector<int> ans(maxc);
        std::vector<double> cost((size_t)maxc * maxc), pxy(2 * (size_t)maxc), cxy(2 * (size_t)maxc);
        WindowSolver solve(solver);
        int64_t counter = 0;
#pragma omp for schedule(dynamic, 8)
        for (int64_t wi = 0; wi < nw; ++wi) {
            if (dl.hit.load(std::memory_order_relaxed)) continue;
            if (dl.active && ((++counter) & 15) == 0 && dl.passed()) continue;
            const Window& win = wins[wi];
            int m = 0;
            for (int j = win.j0; j < win.j1; ++j) {
                const int32_t* grow = pb.grid + (int64_t)(win.b + j * sy) * pb.W;
                for (int i = win.i0; i < win.i1; ++i) {
                    const int32_t id = grow[win.a + i * sx];
                    if (id >= 0) cells[m++] = id;
                }
            }
            int k = 0;
            int64_t maxstamp = -1;
            for (int c = 0; c < m; ++c) {
                const int32_t o = occ[cells[c]];
                oldocc[c] = o;
                if (o >= 0) {
                    rows[k] = o;
                    curcol[k] = c;
                    ++k;
                }
                maxstamp = std::max(maxstamp, stamp[cells[c]]);
            }
            if (k == 0) continue;
            if (skip_clean && last_solved[wi] >= 0 && maxstamp <= last_solved[wi]) continue;
            for (int c = 0; c < m; ++c) {
                cxy[2 * c] = pb.C[2 * (int64_t)cells[c]];
                cxy[2 * c + 1] = pb.C[2 * (int64_t)cells[c] + 1];
            }
            for (int r = 0; r < k; ++r) {
                const double px = pb.P[2 * (int64_t)rows[r]];
                const double py = pb.P[2 * (int64_t)rows[r] + 1];
                pxy[2 * r] = px;
                pxy[2 * r + 1] = py;
                double* crow = cost.data() + (size_t)r * m;
                for (int c = 0; c < m; ++c) {
                    const double dx = px - cxy[2 * c];
                    const double dy = py - cxy[2 * c + 1];
                    crow[c] = dx * dx + dy * dy;
                }
            }
            double cur = 0.0;
            for (int r = 0; r < k; ++r) cur += cost[(size_t)r * m + curcol[r]];
            const double best = solve(cost.data(), k, m, ans.data(), pxy.data(), cxy.data());
            ++solved;
            last_solved[wi] = phase_id;
            // Accept only strict improvements beyond rounding noise; ties never move,
            // which makes the cost strictly decreasing and guarantees termination.
            if (best < cur - tol_rel * cur) {
                for (int c = 0; c < m; ++c) newocc[c] = -1;
                for (int r = 0; r < k; ++r) newocc[ans[r]] = rows[r];
                for (int c = 0; c < m; ++c) {
                    if (newocc[c] != oldocc[c]) {
                        occ[cells[c]] = newocc[c];
                        stamp[cells[c]] = phase_id;
                        if (newocc[c] >= 0) {
                            pos[newocc[c]] = cells[c];
                            ++moved;
                        }
                    }
                }
                ++improved;
                gain += cur - best;
            }
        }
    }
    st.solved = solved;
    st.improved = improved;
    st.moved = moved;
    st.gain = gain;
    return st;
}

static Problem make_problem(const CArr2<double>& P, const CArr2<int32_t>& grid, const CArr2<double>& C) {
    Problem pb;
    if (P.shape(1) != 2 || C.shape(1) != 2) throw std::invalid_argument("P and C must have shape (n, 2)");
    pb.N = (int64_t)P.shape(0);
    pb.M = (int64_t)C.shape(0);
    pb.H = (int)grid.shape(0);
    pb.W = (int)grid.shape(1);
    pb.P = P.data();
    pb.grid = grid.data();
    pb.C = C.data();
    return pb;
}

// Run a ShuffleSnap schedule in place on (occ, pos).
// stages: (S, 4) int32 rows (sx, sy, max_rounds, stop_when_no_moves); max_rounds < 0
//         means unlimited rounds (then stop_when_no_moves should be 1).
// offsets: (K, 2) int32 rows (ox, oy): the phases of one round, in order.
// Returns (trace, finished) where trace is a flat list of rows with columns
//   elapsed_s, cost, stage, round, phase, sx, sy, windows, solved, improved, moved
// and finished is false if the time budget expired before the schedule completed.
static std::pair<std::vector<double>, bool> run_schedule(
    CArr2<double> P, CArr2<int32_t> grid, CArr2<double> C, MArr1<int32_t> occ, MArr1<int32_t> pos,
    CArr2<int32_t> stages, CArr2<int32_t> offsets, int window, double time_budget, int threads,
    double tol_rel, bool skip_clean, int solver) {
    Problem pb = make_problem(P, grid, C);
    if ((int64_t)occ.shape(0) != pb.M || (int64_t)pos.shape(0) != pb.N)
        throw std::invalid_argument("occ must have length M and pos length N");
    if (stages.shape(1) != 4 || offsets.shape(1) != 2) throw std::invalid_argument("bad stages/offsets shape");
    if (window < 1 || window > 32) throw std::invalid_argument("window must be in [1, 32]");
    WindowSolver::check(solver);
    const int nthreads = resolve_threads(threads);
    int32_t* occp = occ.data();
    int32_t* posp = pos.data();
    const int64_t S = (int64_t)stages.shape(0);
    const int64_t K = (int64_t)offsets.shape(0);
    std::vector<int32_t> st_(stages.data(), stages.data() + 4 * S);
    std::vector<int32_t> of_(offsets.data(), offsets.data() + 2 * K);

    std::vector<double> trace;
    bool finished = true;
    {
        nb::gil_scoped_release release;
        const auto t0 = Clock::now();
        Deadline dl;
        // A negative budget means "none" (the Python layer passes -1 for None and rejects
        // other negative values); a zero budget returns the start without running a phase.
        if (time_budget >= 0 && std::isfinite(time_budget)) {
            dl.active = true;
            dl.t = t0 + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(time_budget));
        }
        std::vector<int64_t> stamp(pb.M, -1);
        int64_t phase_id = 0;
        auto record = [&](int64_t s, int64_t r, int64_t k, int sx, int sy, const PhaseStats& ps) {
            const double el = std::chrono::duration<double>(Clock::now() - t0).count();
            const double c = total_cost(pb, posp, nthreads);
            const double row[11] = {el, c, (double)s, (double)r, (double)k, (double)sx, (double)sy,
                                    (double)ps.windows, (double)ps.solved, (double)ps.improved, (double)ps.moved};
            trace.insert(trace.end(), row, row + 11);
        };
        {
            PhaseStats init;
            record(-1, -1, -1, 0, 0, init);
        }
        for (int64_t s = 0; s < S && finished; ++s) {
            const int sx = std::max(1, st_[4 * s]), sy = std::max(1, st_[4 * s + 1]);
            const int64_t max_rounds = st_[4 * s + 2] < 0 ? INT64_MAX : st_[4 * s + 2];
            const bool stop_nm = st_[4 * s + 3] != 0;
            std::vector<std::vector<Window>> wl(K);
            std::vector<std::vector<int64_t>> ls(K);
            for (int64_t k = 0; k < K; ++k) {
                wl[k] = make_windows(pb, sx, sy, window, of_[2 * k], of_[2 * k + 1]);
                ls[k].assign(wl[k].size(), -1);
            }
            for (int64_t r = 0; r < max_rounds; ++r) {
                int64_t round_moves = 0;
                for (int64_t k = 0; k < K; ++k) {
                    if (dl.passed()) {
                        finished = false;
                        break;
                    }
                    PhaseStats ps = run_phase(pb, occp, posp, stamp.data(), wl[k], ls[k], sx, sy, window,
                                              phase_id++, skip_clean, tol_rel, nthreads, dl, solver);
                    round_moves += ps.moved;
                    record(s, r, k, sx, sy, ps);
                    if (dl.hit.load()) {
                        finished = false;
                        break;
                    }
                }
                if (!finished) break;
                if (stop_nm && round_moves == 0) break;
            }
        }
    }
    return {trace, finished};
}

// Recursive bisection start: split the cell set at its median along its longer
// extent, split the points at the proportional rank along the same axis, recurse.
static void init_bisect(CArr2<double> P, CArr2<double> C, MArr1<int32_t> pos) {
    const int64_t N = (int64_t)P.shape(0), M = (int64_t)C.shape(0);
    if (N > M) throw std::invalid_argument("more points than cells");
    if ((int64_t)pos.shape(0) != N) throw std::invalid_argument("pos must have length N");
    const double* p = P.data();
    const double* c = C.data();
    int32_t* out = pos.data();
    nb::gil_scoped_release release;
    std::vector<int32_t> pidx(N), cidx(M);
    std::iota(pidx.begin(), pidx.end(), 0);
    std::iota(cidx.begin(), cidx.end(), 0);
    struct Task {
        int64_t pb, pe, cb, ce;
    };
    std::vector<Task> stack;
    stack.push_back({0, N, 0, M});
    while (!stack.empty()) {
        Task t = stack.back();
        stack.pop_back();
        const int64_t np = t.pe - t.pb, nc = t.ce - t.cb;
        if (np == 0) continue;
        if (nc == 1) {
            out[pidx[t.pb]] = cidx[t.cb];
            continue;
        }
        double xmin = INFINITY, xmax = -INFINITY, ymin = INFINITY, ymax = -INFINITY;
        for (int64_t i = t.cb; i < t.ce; ++i) {
            const double x = c[2 * (int64_t)cidx[i]], y = c[2 * (int64_t)cidx[i] + 1];
            xmin = std::min(xmin, x);
            xmax = std::max(xmax, x);
            ymin = std::min(ymin, y);
            ymax = std::max(ymax, y);
        }
        const int ax = (xmax - xmin) >= (ymax - ymin) ? 0 : 1;
        const int64_t half = nc / 2;
        std::nth_element(cidx.begin() + t.cb, cidx.begin() + t.cb + half, cidx.begin() + t.ce,
                         [&](int32_t u, int32_t v) {
                             const double a0 = c[2 * (int64_t)u + ax], b0 = c[2 * (int64_t)v + ax];
                             if (a0 != b0) return a0 < b0;
                             const double a1 = c[2 * (int64_t)u + 1 - ax], b1 = c[2 * (int64_t)v + 1 - ax];
                             if (a1 != b1) return a1 < b1;
                             return u < v;
                         });
        int64_t kl = (np * half + nc / 2) / nc;
        kl = std::max(kl, np - (nc - half));
        kl = std::min(kl, std::min(np, half));
        if (kl > 0 && kl < np)
            std::nth_element(pidx.begin() + t.pb, pidx.begin() + t.pb + kl, pidx.begin() + t.pe,
                             [&](int32_t u, int32_t v) {
                                 const double a0 = p[2 * (int64_t)u + ax], b0 = p[2 * (int64_t)v + ax];
                                 if (a0 != b0) return a0 < b0;
                                 const double a1 = p[2 * (int64_t)u + 1 - ax], b1 = p[2 * (int64_t)v + 1 - ax];
                                 if (a1 != b1) return a1 < b1;
                                 return u < v;
                             });
        stack.push_back({t.pb, t.pb + kl, t.cb, t.cb + half});
        stack.push_back({t.pb + kl, t.pe, t.cb + half, t.ce});
    }
}

// Exact small LAP, exposed for independent testing of the window solver.
static std::pair<std::vector<int>, double> solve_lap(CArr2<double> cost, int solver) {
    const int k = (int)cost.shape(0), m = (int)cost.shape(1);
    if (k > m) throw std::invalid_argument("need rows <= cols");
    WindowSolver::check(solver);
    std::vector<int> ans(k);
    WindowSolver solve(solver);
    double total = 0.0;
    if (k) total = solve(cost.data(), k, m, ans.data());
    return {ans, total};
}

// One window as the kernel sees it: squared distances from points P (k, 2) to cells C (m, 2),
// solved with the coordinates available to the solver (for the geometric start).
static std::pair<std::vector<int>, double> solve_window(CArr2<double> P, CArr2<double> C, int solver) {
    const int k = (int)P.shape(0), m = (int)C.shape(0);
    if (P.shape(1) != 2 || C.shape(1) != 2) throw std::invalid_argument("P and C must have shape (n, 2)");
    if (k > m) throw std::invalid_argument("need points <= cells");
    WindowSolver::check(solver);
    std::vector<double> cost((size_t)k * m);
    for (int r = 0; r < k; ++r)
        for (int c = 0; c < m; ++c) {
            const double dx = P.data()[2 * r] - C.data()[2 * c], dy = P.data()[2 * r + 1] - C.data()[2 * c + 1];
            cost[(size_t)r * m + c] = dx * dx + dy * dy;
        }
    std::vector<int> ans(k);
    WindowSolver solve(solver);
    double total = 0.0;
    if (k) total = solve(cost.data(), k, m, ans.data(), P.data(), C.data());
    return {ans, total};
}

// Time `reps` solves of each cost matrix in a batch (solver micro-benchmark).
static double bench_lap(CArr1<double> costs, int k, int m, int count, int reps, int solver) {
    WindowSolver::check(solver);
    std::vector<int> ans(k);
    WindowSolver solve(solver);
    double sink = 0.0;
    const auto t0 = Clock::now();
    for (int r = 0; r < reps; ++r)
        for (int c = 0; c < count; ++c) {
            const double* a = costs.data() + (size_t)c * k * m;
            sink += solve(a, k, m, ans.data());
        }
    const double el = std::chrono::duration<double>(Clock::now() - t0).count();
    return sink == -1.0 ? -1.0 : el;
}

static double assignment_cost(CArr2<double> P, CArr2<double> C, CArr1<int32_t> pos, int threads) {
    Problem pb;
    pb.N = (int64_t)P.shape(0);
    pb.M = (int64_t)C.shape(0);
    pb.P = P.data();
    pb.C = C.data();
    return total_cost(pb, pos.data(), resolve_threads(threads));
}

// Number of windows per phase for a stride/offset (for documentation and tests).
static std::vector<std::vector<int32_t>> list_windows(CArr2<int32_t> grid, int sx, int sy, int w, int ox, int oy) {
    Problem pb;
    pb.H = (int)grid.shape(0);
    pb.W = (int)grid.shape(1);
    pb.grid = grid.data();
    auto wins = make_windows(pb, sx, sy, w, ox, oy);
    std::vector<std::vector<int32_t>> out;
    out.reserve(wins.size());
    for (auto& x : wins) out.push_back({x.a, x.b, x.i0, x.i1, x.j0, x.j1});
    return out;
}

static int omp_threads() { return resolve_threads(0); }

template <typename T>
static nb::ndarray<nb::numpy, T, nb::ndim<1>> to_numpy(std::vector<T>&& vec) {
    auto* p = new std::vector<T>(std::move(vec));
    nb::capsule owner(p, [](void* q) noexcept { delete static_cast<std::vector<T>*>(q); });
    return nb::ndarray<nb::numpy, T, nb::ndim<1>>(p->data(), {p->size()}, owner);
}

// Global Lagrangian lower bound for cell potentials v (see _dual.hpp).
// Returns (lb, u, argmin, viol_i, viol_j): u[i] = min_j (c_ij - v_j) over ALL usable
// cells, argmin[i] the minimizing cell, and up to max_viol cells per point whose
// reduced cost is below that of the point's current cell pos[i] by more than eps.
static nb::tuple dual_bound(CArr2<double> P, CArr2<int32_t> grid, CArr2<double> C, CArr1<double> v,
                            CArr1<int32_t> pos, double eps, int max_viol, int block, int threads) {
    Problem pb = make_problem(P, grid, C);
    if ((int64_t)v.shape(0) != pb.M || (int64_t)pos.shape(0) != pb.N) throw std::invalid_argument("bad v/pos length");
    const int nthreads = resolve_threads(threads);
    const double* vp = v.data();
    const int32_t* posp = pos.data();
    std::vector<double> u(pb.N);
    std::vector<int32_t> arg(pb.N);
    std::vector<int32_t> vi, vj;
    double vmax = -INFINITY, vsum = 0.0;
    for (int64_t j = 0; j < pb.M; ++j) {
        vmax = std::max(vmax, vp[j]);
        vsum += vp[j];
    }
    {
        nb::gil_scoped_release release;
        BlockIndex bi = build_blocks(pb.grid, pb.W, pb.H, pb.C, vp, std::max(1, block));
        std::vector<std::vector<int32_t>> tvi(nthreads), tvj(nthreads);
#pragma omp parallel num_threads(nthreads)
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            std::vector<std::pair<double, int32_t>> viol;
            std::vector<double> lbs;
#pragma omp for schedule(dynamic, 256)
            for (int64_t i = 0; i < pb.N; ++i) {
                const double px = pb.P[2 * i], py = pb.P[2 * i + 1];
                const int32_t o = posp[i];
                const double dx = px - pb.C[2 * (int64_t)o], dy = py - pb.C[2 * (int64_t)o + 1];
                const double own = dx * dx + dy * dy - vp[o];
                viol.clear();
                PointDual pd;
                point_min_reduced(bi, pb.C, vp, px, py, own, o, pd, own - eps, max_viol, viol, lbs);
                u[i] = pd.u;
                arg[i] = pd.arg;
                for (auto& q : viol) {
                    if (q.second == o) continue;
                    tvi[tid].push_back((int32_t)i);
                    tvj[tid].push_back(q.second);
                }
            }
        }
        for (int t = 0; t < nthreads; ++t) {
            vi.insert(vi.end(), tvi[t].begin(), tvi[t].end());
            vj.insert(vj.end(), tvj[t].begin(), tvj[t].end());
        }
    }
    // Deterministic summation.
    double usum = 0.0;
    for (int64_t i = 0; i < pb.N; ++i) usum += u[i];
    const double lb = usum + vsum - (double)(pb.M - pb.N) * (pb.M > pb.N ? vmax : 0.0);
    return nb::make_tuple(lb, to_numpy(std::move(u)), to_numpy(std::move(arg)), to_numpy(std::move(vi)),
                          to_numpy(std::move(vj)));
}

// Shortest-path potentials on the "cell graph" of a matching: arcs a->b with integer
// weights.  Returns (ok, d) with d[b] <= d[a] + w for every arc; ok is false if a
// negative cycle exists (the matching is not optimal on the arc set).
static nb::tuple spfa(int64_t n, CArr1<int32_t> tail, CArr1<int32_t> head, CArr1<int64_t> w) {
    const int64_t E = (int64_t)tail.shape(0);
    if ((int64_t)head.shape(0) != E || (int64_t)w.shape(0) != E) throw std::invalid_argument("arc arrays differ");
    std::vector<int64_t> start(n + 1, 0);
    std::vector<int32_t> hd(E);
    std::vector<int64_t> ww(E);
    std::vector<int64_t> d;
    bool ok;
    {
        nb::gil_scoped_release release;
        const int32_t* tp = tail.data();
        const int32_t* hp = head.data();
        const int64_t* wp = w.data();
        for (int64_t e = 0; e < E; ++e) {
            if (tp[e] < 0 || tp[e] >= n || hp[e] < 0 || hp[e] >= n) throw std::invalid_argument("arc out of range");
            start[tp[e] + 1]++;
        }
        for (int64_t a = 0; a < n; ++a) start[a + 1] += start[a];
        std::vector<int64_t> fill(start.begin(), start.end() - 1);
        for (int64_t e = 0; e < E; ++e) {
            const int64_t k = fill[tp[e]]++;
            hd[k] = hp[e];
            ww[k] = wp[e];
        }
        ok = spfa_potentials(n, start, hd, ww, d);
    }
    return nb::make_tuple(ok, to_numpy(std::move(d)));
}

}  // namespace shufflesnap

NB_MODULE(_core, m) {
    using namespace shufflesnap;
    m.doc() = "ShuffleSnap compiled kernels";
    m.def("run_schedule", &run_schedule, "P"_a, "grid"_a, "C"_a, "occ"_a, "pos"_a, "stages"_a, "offsets"_a,
          "window"_a = 6, "time_budget"_a = -1.0, "threads"_a = 0, "tol_rel"_a = 1e-12, "skip_clean"_a = true,
          "solver"_a = 0);
    m.def("init_bisect", &init_bisect, "P"_a, "C"_a, "pos"_a);
    m.def("solve_lap", &solve_lap, "cost"_a, "solver"_a = 0);
    m.def("solve_window", &solve_window, "P"_a, "C"_a, "solver"_a = 0);
    m.def("bench_lap", &bench_lap, "costs"_a, "k"_a, "m"_a, "count"_a, "reps"_a, "solver"_a);
    m.def("assignment_cost", &assignment_cost, "P"_a, "C"_a, "pos"_a, "threads"_a = 0);
    m.def("list_windows", &list_windows, "grid"_a, "sx"_a, "sy"_a, "window"_a, "ox"_a, "oy"_a);
    m.def("max_threads", &omp_threads);
    m.def("dual_bound", &dual_bound, "P"_a, "grid"_a, "C"_a, "v"_a, "pos"_a, "eps"_a = 0.0, "max_viol"_a = 0,
          "block"_a = 16, "threads"_a = 0);
    m.def("spfa", &spfa, "n"_a, "tail"_a, "head"_a, "w"_a);
#ifdef _OPENMP
    m.attr("has_openmp") = true;
#else
    m.attr("has_openmp") = false;
#endif
}
