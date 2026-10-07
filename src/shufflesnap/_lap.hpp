// Small dense rectangular linear assignment solvers used inside ShuffleSnap windows.
//
// All solvers take a row-major cost matrix with k rows (occupants) and m columns
// (cells), k <= m, and return, for every row, the column it is assigned to so that
// every row gets a distinct column and the total cost is minimal.
//
// The core solver is the O(k^2 m) shortest-augmenting-path Hungarian method with
// row/column potentials (Kuhn-Munkres in the Jonker-Volgenant/Dijkstra form).  It is
// exact for real-valued costs up to floating point rounding.  lap_hungarian_greedy adds a
// greedy warm start; lap_auction uses an eps-scaled auction only to warm-start the same
// shortest-path method, so it is exact too; lap_jv is classic JV.  The Python API defaults
// to lap_geometric, which builds its starting duals from the window's geometry (see there);
// the raw bindings' solver=0 selects plain lap_hungarian.
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace shufflesnap {

struct LapWork {
    std::vector<double> u, v, minv;
    std::vector<int> p, way;
    std::vector<char> used;
    void ensure(int k, int m) {
        if ((int)u.size() < k + 1) u.resize(k + 1);
        if ((int)v.size() < m + 1) {
            v.resize(m + 1);
            minv.resize(m + 1);
            p.resize(m + 1);
            way.resize(m + 1);
            used.resize(m + 1);
        }
    }
};

// Hungarian method (shortest augmenting paths with potentials), k <= m.
// Returns the optimal total cost; row2col[r] receives the column of row r.
inline double lap_hungarian(const double* a, int k, int m, int* row2col, LapWork& w) {
    const double INF = std::numeric_limits<double>::infinity();
    w.ensure(k, m);
    double* u = w.u.data();
    double* v = w.v.data();
    double* minv = w.minv.data();
    int* p = w.p.data();
    int* way = w.way.data();
    char* used = w.used.data();
    std::fill(u, u + k + 1, 0.0);
    std::fill(v, v + m + 1, 0.0);
    std::fill(p, p + m + 1, 0);
    std::fill(way, way + m + 1, 0);
    for (int i = 1; i <= k; ++i) {
        p[0] = i;
        int j0 = 0;
        std::fill(minv, minv + m + 1, INF);
        std::fill(used, used + m + 1, 0);
        do {
            used[j0] = 1;
            const int i0 = p[j0];
            const double* arow = a + (size_t)(i0 - 1) * m;
            const double ui0 = u[i0];
            double delta = INF;
            int j1 = 0;
            for (int j = 1; j <= m; ++j) {
                if (used[j]) continue;
                const double cur = arow[j - 1] - ui0 - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= m; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    for (int j = 1; j <= m; ++j)
        if (p[j]) row2col[p[j] - 1] = j - 1;
    double total = 0.0;
    for (int r = 0; r < k; ++r) total += a[(size_t)r * m + row2col[r]];
    return total;
}


// Hungarian method with a greedy warm start: row reduction u_i = min_j a_ij (v = 0,
// which keeps unmatched columns dual-feasible for k < m), then every row whose
// minimizing column is still free is matched on its zero reduced-cost edge; only the
// remaining rows are augmented by shortest paths.  Same optimum as lap_hungarian.
inline double lap_hungarian_greedy(const double* a, int k, int m, int* row2col, LapWork& w) {
    const double INF = std::numeric_limits<double>::infinity();
    w.ensure(k, m);
    double* u = w.u.data();
    double* v = w.v.data();
    double* minv = w.minv.data();
    int* p = w.p.data();
    int* way = w.way.data();
    char* used = w.used.data();
    std::fill(v, v + m + 1, 0.0);
    std::fill(p, p + m + 1, 0);
    std::fill(way, way + m + 1, 0);
    u[0] = 0.0;
    // row reduction + greedy matching on zero reduced-cost edges
    static thread_local std::vector<int> todo;
    todo.clear();
    for (int i = 1; i <= k; ++i) {
        const double* arow = a + (size_t)(i - 1) * m;
        double mn = arow[0];
        int jm = 0;
        for (int j = 1; j < m; ++j)
            if (arow[j] < mn) {
                mn = arow[j];
                jm = j;
            }
        u[i] = mn;
        if (p[jm + 1] == 0)
            p[jm + 1] = i;
        else {
            todo.push_back(i);
        }
    }
    for (int i : todo) {
        p[0] = i;
        int j0 = 0;
        std::fill(minv, minv + m + 1, INF);
        std::fill(used, used + m + 1, 0);
        do {
            used[j0] = 1;
            const int i0 = p[j0];
            const double* arow = a + (size_t)(i0 - 1) * m;
            const double ui0 = u[i0];
            double delta = INF;
            int j1 = 0;
            for (int j = 1; j <= m; ++j) {
                if (used[j]) continue;
                const double cur = arow[j - 1] - ui0 - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= m; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    for (int j = 1; j <= m; ++j)
        if (p[j]) row2col[p[j] - 1] = j - 1;
    double total = 0.0;
    for (int r = 0; r < k; ++r) total += a[(size_t)r * m + row2col[r]];
    return total;
}

// Auction (Bertsekas 1979; forward Gauss-Seidel bidding with eps-scaling) followed by an
// exact finish.  Rectangular problems are padded to m x m with m - k zero-cost rows.
// The auction ends eps-optimal, not optimal, so its prices seed a Hungarian finish:
//   1. tighten: lower each assigned column's price until its row's auction edge ties
//      the row's best alternative (a heuristic: it can untighten other rows' edges);
//   2. set v_j = -price_j and u_i = min_j (c_ij - v_j), a feasible dual; each row keeps
//      its auction column if that edge is tight, else takes a free tight column;
//   3. augment the remaining rows along shortest paths as in lap_hungarian.
// The result is exact like lap_hungarian; the auction only supplies a warm start.
struct AuctionWork {
    std::vector<double> price, zero, u, v, minv;
    std::vector<int> owner, assigned, queue, p, way, todo;
    std::vector<char> used;
    void ensure(int m) {
        if ((int)price.size() < m + 1) {
            price.resize(m + 1);
            zero.assign(m + 1, 0.0);
            u.resize(m + 1);
            v.resize(m + 1);
            minv.resize(m + 1);
            owner.resize(m + 1);
            assigned.resize(m + 1);
            queue.resize(m + 1);
            p.resize(m + 1);
            way.resize(m + 1);
            used.resize(m + 1);
        }
    }
};

// eps schedule, relative to the cost range: tuned on real 8x8 window matrices
// (MNIST, uniform and Gaussian-mixture embeddings, balanced preset).
constexpr double AUCTION_EPS0 = 0.05, AUCTION_EPSF = 1e-3, AUCTION_THETA = 8.0;

inline double lap_auction(const double* a, int k, int m, int* row2col, AuctionWork& w) {
    const double INF = std::numeric_limits<double>::infinity();
    const int n = m;
    w.ensure(n);
    double* price = w.price.data();
    int* owner = w.owner.data();
    int* asg = w.assigned.data();
    int* q = w.queue.data();
    auto row = [&](int i) -> const double* { return i < k ? a + (size_t)i * m : w.zero.data(); };
    double cmin = INF, cmax = -INF;
    for (size_t t = 0; t < (size_t)k * m; ++t) {
        cmin = std::min(cmin, a[t]);
        cmax = std::max(cmax, a[t]);
    }
    if (k < m) cmin = std::min(cmin, 0.0), cmax = std::max(cmax, 0.0);
    const double range = cmax - cmin;
    std::fill(price, price + n, 0.0);
    for (int i = 0; i < n; ++i) asg[i] = i;
    if (range > 0 && n > 1) {
        double eps = range * AUCTION_EPS0;
        const double epsf = range * AUCTION_EPSF;
        for (;;) {
            std::fill(owner, owner + n, -1);
            for (int i = 0; i < n; ++i) q[i] = i;  // circular queue, full: head == tail
            int qh = 0, qt = 0, nq = n;
            while (nq) {
                const int i = q[qh];
                qh = qh + 1 == n ? 0 : qh + 1;
                --nq;
                const double* r = row(i);
                double b1 = INF, b2 = INF;
                int j1 = 0;
                for (int j = 0; j < n; ++j) {
                    const double x = r[j] + price[j];
                    if (x < b2) {
                        if (x < b1) {
                            b2 = b1;
                            b1 = x;
                            j1 = j;
                        } else {
                            b2 = x;
                        }
                    }
                }
                price[j1] += (b2 - b1) + eps;
                const int prev = owner[j1];
                owner[j1] = i;
                asg[i] = j1;
                if (prev >= 0) {
                    q[qt] = prev;
                    qt = qt + 1 == n ? 0 : qt + 1;
                    ++nq;
                }
            }
            if (eps <= epsf) break;
            eps = std::max(eps / AUCTION_THETA, epsf);
        }
    }
    // 1. tighten
    for (int i = 0; i < n; ++i) {
        const double* r = row(i);
        const int ja = asg[i];
        double m2 = INF;
        for (int j = 0; j < n; ++j)
            if (j != ja) m2 = std::min(m2, r[j] + price[j]);
        if (r[ja] + price[ja] > m2) price[ja] = m2 - r[ja];
    }
    // 2. feasible dual and tight initial matching (1-based arrays as in lap_hungarian)
    double* u = w.u.data();
    double* v = w.v.data();
    double* minv = w.minv.data();
    int* p = w.p.data();
    int* way = w.way.data();
    char* used = w.used.data();
    v[0] = 0.0;
    std::fill(p, p + n + 1, 0);
    std::fill(way, way + n + 1, 0);
    for (int j = 0; j < n; ++j) v[j + 1] = -price[j];
    w.todo.clear();
    for (int i = 0; i < n; ++i) {
        const double* r = row(i);
        double mn = INF;
        int jm = 0;
        for (int j = 0; j < n; ++j) {
            const double x = r[j] + price[j];
            if (x < mn) mn = x, jm = j;
        }
        u[i + 1] = mn;
        const int ja = asg[i];
        if (r[ja] + price[ja] <= mn && p[ja + 1] == 0)
            p[ja + 1] = i + 1;
        else if (p[jm + 1] == 0)
            p[jm + 1] = i + 1;
        else
            w.todo.push_back(i + 1);
    }
    // 3. shortest augmenting paths for the unmatched rows
    for (int i : w.todo) {
        p[0] = i;
        int j0 = 0;
        std::fill(minv, minv + n + 1, INF);
        std::fill(used, used + n + 1, 0);
        do {
            used[j0] = 1;
            const int i0 = p[j0];
            const double* arow = row(i0 - 1);
            const double ui0 = u[i0];
            double delta = INF;
            int j1 = 0;
            for (int j = 1; j <= n; ++j) {
                if (used[j]) continue;
                const double cur = arow[j - 1] - ui0 - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    for (int j = 1; j <= n; ++j)
        if (p[j] && p[j] <= k) row2col[p[j] - 1] = j - 1;
    double total = 0.0;
    for (int r = 0; r < k; ++r) total += a[(size_t)r * m + row2col[r]];
    return total;
}

// Geometric window solver for squared-distance costs c_ij = |p_i - q_j|^2 between k points
// and m cells.  Exact like lap_hungarian; the geometry only builds the starting duals.
//   1. Quantile start: the optimal cell duals of a squared-distance transport problem are a
//      discretized Brenier potential, v(q) = |q|^2 - 2 psi(q) with psi convex, whose gradient
//      map sends points to cells.  Per axis, the 1D optimal map (sorted points to sorted
//      cells) has such a potential: psi' between consecutive distinct cell coordinates is
//      the point quantile at that cumulative cell mass.  The start is the separable sum of
//      the two axes' potentials, computed in window-centred coordinates.  It removes the
//      bulk offset and spread mismatch that makes clustered windows hard: every point of a
//      far-away cluster otherwise prefers the same corner cell.
//   2. Row reduction u_i = min_j (c_ij - v_j), greedy matching on each row's argmin, then a
//      column reduction v_j = min_i (c_ij - u_i) (both keep the duals feasible).
//   3. Kuhn's augmenting paths restricted to tight edges for the rows still free.
//   4. Shortest augmenting paths (Crouse 2016 form: Dijkstra over a shrinking list of
//      unscanned columns, lazy dual updates, ties broken towards free columns).
// Rectangular problems (k < m) are padded to m x m with zero-cost rows when they use the
// geometric start, since arbitrary starting duals are only valid for the square problem
// (see GEO_RECT_COLLIDE).  Without geometry (P or C null) the start is v = 0.
struct GeoWork {
    std::vector<double> u, v, spc, zero, ps, cs, cv, psi;
    std::vector<int> path, row4col, col4row, rem, srows, todo, rest, seen, stk_row, stk_cur;
    void ensure(int m) {
        if ((int)u.size() < m) {
            u.resize(m);
            v.resize(m);
            spc.resize(m);
            zero.assign(m, 0.0);
            ps.resize(m);
            cs.resize(m);
            path.resize(m);
            row4col.resize(m);
            col4row.resize(m);
            rem.resize(m);
            srows.resize(m);
            seen.resize(m);
        }
    }
};

// v_j = sum over axes of (q_j^2 - 2 psi(q_j)), coordinates centred on the cells' mean.
inline void geo_quantile_duals(const double* P, const double* C, int k, int m, double* v, GeoWork& w) {
    double c0[2] = {0.0, 0.0};
    for (int j = 0; j < m; ++j) c0[0] += C[2 * j], c0[1] += C[2 * j + 1];
    c0[0] /= m, c0[1] /= m;
    std::fill(v, v + m, 0.0);
    double* ps = w.ps.data();
    double* cs = w.cs.data();
    for (int t = 0; t < 2; ++t) {
        for (int i = 0; i < k; ++i) ps[i] = P[2 * i + t] - c0[t];
        for (int j = 0; j < m; ++j) cs[j] = C[2 * j + t] - c0[t];
        std::sort(ps, ps + k);
        std::sort(cs, cs + m);
        w.cv.clear();
        w.psi.clear();
        double acc = 0.0;
        for (int j = 0; j < m;) {
            int e = j;
            while (e < m && cs[e] == cs[j]) ++e;
            if (!w.cv.empty()) {
                // point quantile at cumulative cell mass j / m (linear interpolation)
                const double pos = (double)j / m * k - 0.5;
                double q;
                if (pos <= 0.0) {
                    q = ps[0];
                } else if (pos >= k - 1) {
                    q = ps[k - 1];
                } else {
                    const int lo = (int)pos;
                    q = ps[lo] + (pos - lo) * (ps[lo + 1] - ps[lo]);
                }
                acc += (cs[j] - w.cv.back()) * q;
            }
            w.cv.push_back(cs[j]);
            w.psi.push_back(acc);
            j = e;
        }
        for (int j = 0; j < m; ++j) {
            const double c = C[2 * j + t] - c0[t];
            const size_t idx = std::lower_bound(w.cv.begin(), w.cv.end(), c) - w.cv.begin();
            v[j] += c * c - 2.0 * w.psi[idx];
        }
    }
}

// One shortest augmenting path from free row `cur`, then the dual update and augmentation.
template <class Row>
inline void geo_augment(Row row, int m, int cur, GeoWork& w) {
    const double INF = std::numeric_limits<double>::infinity();
    double* u = w.u.data();
    double* v = w.v.data();
    double* spc = w.spc.data();
    int* path = w.path.data();
    int* row4col = w.row4col.data();
    int* col4row = w.col4row.data();
    int* rem = w.rem.data();
    int* srows = w.srows.data();
    for (int j = 0; j < m; ++j) {
        rem[j] = j;
        spc[j] = INF;
    }
    int nrem = m, nsr = 0, i = cur, sink = -1;
    double minVal = 0.0;
    while (sink < 0) {
        srows[nsr++] = i;
        const double* r = row(i);
        const double ui = u[i];
        double lowest = INF;
        int index = 0;
        for (int it = 0; it < nrem; ++it) {
            const int j = rem[it];
            const double rr = minVal + r[j] - ui - v[j];
            if (rr < spc[j]) {
                path[j] = i;
                spc[j] = rr;
            }
            if (spc[j] < lowest || (spc[j] == lowest && row4col[j] < 0)) {
                lowest = spc[j];
                index = it;
            }
        }
        minVal = lowest;
        const int j = rem[index];
        if (row4col[j] < 0)
            sink = j;
        else
            i = row4col[j];
        rem[index] = rem[--nrem];
        rem[nrem] = j;  // scanned columns collect at the tail
    }
    u[cur] += minVal;
    for (int s = 1; s < nsr; ++s) u[srows[s]] += minVal - spc[col4row[srows[s]]];
    for (int it = nrem; it < m; ++it) v[rem[it]] -= minVal - spc[rem[it]];
    for (int j = sink;;) {
        const int ii = path[j];
        row4col[j] = ii;
        const int next = col4row[ii];
        col4row[ii] = j;
        if (ii == cur) break;
        j = next;
    }
}

// Windows with empty cells (k < m) are solved as the rectangular problem from v = 0 when
// they look uniform (few rows collide on their cheapest cell): padding them would add one
// shortest-path search per empty cell.  Clustered ones are padded and use the geometric start.
constexpr double GEO_RECT_COLLIDE = 0.5;

inline double lap_geometric(const double* a, int k, int m, int* row2col, GeoWork& w, const double* P = nullptr,
                            const double* C = nullptr) {
    const double INF = std::numeric_limits<double>::infinity();
    w.ensure(m);
    double* u = w.u.data();
    double* v = w.v.data();
    int* row4col = w.row4col.data();
    int* col4row = w.col4row.data();
    auto row = [&](int i) -> const double* { return i < k ? a + (size_t)i * m : w.zero.data(); };
    // 2. row reduction u_i = min_j (c_ij - v_j) and greedy matching on each row's argmin
    auto greedy = [&](int n) {
        std::fill(row4col, row4col + m, -1);
        w.todo.clear();
        for (int i = 0; i < n; ++i) {
            const double* r = row(i);
            double mn = INF;
            int jm = 0;
            for (int j = 0; j < m; ++j) {
                const double x = r[j] - v[j];
                if (x < mn) mn = x, jm = j;
            }
            u[i] = mn;
            if (row4col[jm] < 0) {
                row4col[jm] = i;
                col4row[i] = jm;
            } else {
                col4row[i] = -1;
                w.todo.push_back(i);
            }
        }
    };
    const bool geometry = P && C && k > 0;
    int n = m;  // rows: square or padded
    if (k < m) {
        std::fill(v, v + m, 0.0);
        greedy(k);
        if (!geometry || (double)w.todo.size() <= GEO_RECT_COLLIDE * k) n = k;
    }
    if (n == m) {
        // 1. quantile start (square or padded)
        if (geometry)
            geo_quantile_duals(P, C, k, m, v, w);
        else
            std::fill(v, v + m, 0.0);
        greedy(n);
        // column reduction v_j = min_i (c_ij - u_i): keeps the duals feasible, never lowers
        // v, and leaves the greedy edges tight (each matched row attains its column's min)
        if (!w.todo.empty())
            for (int j = 0; j < m; ++j) {
                double mn = INF;
                for (int i = 0; i < n; ++i) mn = std::min(mn, row(i)[j] - u[i]);
                v[j] = mn;
            }
    }
    if (!w.todo.empty()) {
        // 3. Kuhn's augmenting paths on tight edges (iterative DFS, one visit per column)
        int* seen = w.seen.data();
        std::fill(seen, seen + m, -1);
        w.rest.clear();
        for (int i0 : w.todo) {
            w.stk_row.assign(1, i0);
            w.stk_cur.assign(1, 0);
            bool found = false;
            while (!w.stk_row.empty() && !found) {
                const int i = w.stk_row.back();
                const double* r = row(i);
                bool pushed = false;
                for (int& j = w.stk_cur.back(); j < m; ++j) {
                    if (seen[j] == i0 || r[j] - u[i] - v[j] > 0.0) continue;
                    seen[j] = i0;
                    if (row4col[j] < 0) {
                        for (int s = (int)w.stk_row.size() - 1, jj = j; s >= 0; --s) {
                            const int ii = w.stk_row[s];
                            const int old = col4row[ii];
                            row4col[jj] = ii;
                            col4row[ii] = jj;
                            jj = old;
                        }
                        found = true;
                        break;
                    }
                    const int next = row4col[j];
                    ++j;
                    w.stk_row.push_back(next);
                    w.stk_cur.push_back(0);
                    pushed = true;
                    break;
                }
                if (!found && !pushed) {
                    w.stk_row.pop_back();
                    w.stk_cur.pop_back();
                }
            }
            if (!found) w.rest.push_back(i0);
        }
        // 4. shortest augmenting paths for the rest
        for (int i : w.rest) geo_augment(row, m, i, w);
    }
    double total = 0.0;
    for (int r = 0; r < k; ++r) {
        row2col[r] = col4row[r];
        total += a[(size_t)r * m + col4row[r]];
    }
    return total;
}

// Jonker-Volgenant (1987) dense LAP: column reduction, reduction transfer, two rounds of
// augmenting row reduction, then Dijkstra-style augmentation.  Rectangular problems
// (k < m) are solved as m x m with m - k virtual zero-cost rows.  Exact for real costs.
struct JvWork {
    std::vector<double> v, d;
    std::vector<int> rowsol, colsol, freer, collist, pred, matches;
    void ensure(int m) {
        if ((int)v.size() < m) {
            v.resize(m);
            d.resize(m);
            rowsol.resize(m);
            colsol.resize(m);
            freer.resize(m);
            collist.resize(m);
            pred.resize(m);
            matches.resize(m);
        }
    }
};

inline double lap_jv(const double* a, int k, int m, int* row2col, JvWork& w) {
    const double BIG = std::numeric_limits<double>::infinity();
    const int n = m;
    w.ensure(n);
    double* v = w.v.data();
    double* d = w.d.data();
    int* rowsol = w.rowsol.data();
    int* colsol = w.colsol.data();
    int* freer = w.freer.data();
    int* collist = w.collist.data();
    int* pred = w.pred.data();
    int* matches = w.matches.data();
    auto cost = [&](int i, int j) -> double { return i < k ? a[(size_t)i * m + j] : 0.0; };
    std::fill(matches, matches + n, 0);
    // column reduction
    for (int j = n - 1; j >= 0; --j) {
        double mn = cost(0, j);
        int imin = 0;
        for (int i = 1; i < n; ++i) {
            const double c = cost(i, j);
            if (c < mn) {
                mn = c;
                imin = i;
            }
        }
        v[j] = mn;
        if (++matches[imin] == 1) {
            rowsol[imin] = j;
            colsol[j] = imin;
        } else if (v[j] < v[rowsol[imin]]) {
            const int j1 = rowsol[imin];
            rowsol[imin] = j;
            colsol[j] = imin;
            colsol[j1] = -1;
        } else {
            colsol[j] = -1;
        }
    }
    // reduction transfer
    int numfree = 0;
    for (int i = 0; i < n; ++i) {
        if (matches[i] == 0) {
            freer[numfree++] = i;
        } else if (matches[i] == 1) {
            const int j1 = rowsol[i];
            double mn = BIG;
            for (int j = 0; j < n; ++j)
                if (j != j1) mn = std::min(mn, cost(i, j) - v[j]);
            if (mn < BIG) v[j1] -= mn;
        }
    }
    // augmenting row reduction (two passes); the guard bounds rare float ping-pong
    for (int loop = 0; loop < 2; ++loop) {
        int kk = 0;
        const int prvnumfree = numfree;
        numfree = 0;
        long guard = 0;
        const long guard_max = 8L * n * n;
        while (kk < prvnumfree) {
            const int i = freer[kk++];
            double umin = cost(i, 0) - v[0], usubmin = BIG;
            int j1 = 0, j2 = -1;
            for (int j = 1; j < n; ++j) {
                const double h = cost(i, j) - v[j];
                if (h < usubmin) {
                    if (h >= umin) {
                        usubmin = h;
                        j2 = j;
                    } else {
                        usubmin = umin;
                        umin = h;
                        j2 = j1;
                        j1 = j;
                    }
                }
            }
            int i0 = colsol[j1];
            const bool strict = umin < usubmin;
            if (strict) {
                v[j1] -= (usubmin - umin);
            } else if (i0 > -1 && j2 >= 0) {
                j1 = j2;
                i0 = colsol[j2];
            }
            rowsol[i] = j1;
            colsol[j1] = i;
            if (i0 > -1) {
                if (strict && ++guard < guard_max)
                    freer[--kk] = i0;
                else
                    freer[numfree++] = i0;
            }
        }
    }
    // augmentation
    for (int f = 0; f < numfree; ++f) {
        const int freerow = freer[f];
        for (int j = n - 1; j >= 0; --j) {
            d[j] = cost(freerow, j) - v[j];
            pred[j] = freerow;
            collist[j] = j;
        }
        int low = 0, up = 0, last = 0, endofpath = -1;
        bool found = false;
        double mn = 0.0;
        do {
            if (up == low) {
                last = low - 1;
                mn = d[collist[up++]];
                for (int kx = up; kx < n; ++kx) {
                    const int j = collist[kx];
                    const double h = d[j];
                    if (h <= mn) {
                        if (h < mn) {
                            up = low;
                            mn = h;
                        }
                        collist[kx] = collist[up];
                        collist[up++] = j;
                    }
                }
                for (int kx = low; kx < up; ++kx)
                    if (colsol[collist[kx]] < 0) {
                        endofpath = collist[kx];
                        found = true;
                        break;
                    }
            }
            if (!found) {
                const int j1 = collist[low++];
                const int i = colsol[j1];
                const double h = cost(i, j1) - v[j1] - mn;
                for (int kx = up; kx < n; ++kx) {
                    const int j = collist[kx];
                    const double v2 = cost(i, j) - v[j] - h;
                    if (v2 < d[j]) {
                        pred[j] = i;
                        if (v2 == mn) {
                            if (colsol[j] < 0) {
                                endofpath = j;
                                found = true;
                                break;
                            }
                            collist[kx] = collist[up];
                            collist[up++] = j;
                        }
                        d[j] = v2;
                    }
                }
            }
        } while (!found);
        for (int kx = 0; kx <= last; ++kx) {
            const int j1 = collist[kx];
            v[j1] += d[j1] - mn;
        }
        int i;
        do {
            i = pred[endofpath];
            colsol[endofpath] = i;
            const int j1 = endofpath;
            endofpath = rowsol[i];
            rowsol[i] = j1;
        } while (i != freerow);
    }
    double total = 0.0;
    for (int r = 0; r < k; ++r) {
        row2col[r] = rowsol[r];
        total += a[(size_t)r * m + rowsol[r]];
    }
    return total;
}

}  // namespace shufflesnap
