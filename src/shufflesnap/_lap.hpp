// Small dense rectangular linear assignment solvers used inside ShuffleSnap windows.
//
// All solvers take a row-major cost matrix with k rows (occupants) and m columns
// (cells), k <= m, and return, for every row, the column it is assigned to so that
// every row gets a distinct column and the total cost is minimal.
//
// The default solver is the O(k^2 m) shortest-augmenting-path Hungarian method with
// row/column potentials (Kuhn-Munkres in the Jonker-Volgenant/Dijkstra form).  It is
// exact for real-valued costs up to floating point rounding.
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
