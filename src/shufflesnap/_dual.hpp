// Duality tools used to certify assignments.
//
// For cell potentials v (any real vector) the Lagrangian dual of the assignment LP
//     min sum_ij c_ij x_ij,  sum_j x_ij = 1 (points),  sum_i x_ij <= 1 (cells)
// gives the lower bound
//     LB(v) = sum_i min_j (c_ij - v_j) + sum_j v_j - (M - N) * max_j v_j
// on the optimal total cost (the last term makes the potentials feasible, v <= 0 after a
// shift, when there are more cells than points).  With c_ij the squared distance between
// point i and cell j the inner minimum is found exactly with a block branch-and-bound:
// c_ij - v_j >= dist^2(p_i, box(B)) - max_{j in B} v_j for every block B of cells.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <stdexcept>
#include <vector>

namespace shufflesnap {

struct BlockIndex {
    int W = 0, H = 0, B = 16, nbx = 0, nby = 0;
    std::vector<double> bxmin, bxmax, bymin, bymax, bvmax;  // per block: center box + max potential
    std::vector<int32_t> start, cells;                      // CSR list of cells per block
};

inline BlockIndex build_blocks(const int32_t* grid, int W, int H, const double* C, const double* v, int B) {
    BlockIndex bi;
    bi.W = W;
    bi.H = H;
    bi.B = B;
    bi.nbx = (W + B - 1) / B;
    bi.nby = (H + B - 1) / B;
    const int nb = bi.nbx * bi.nby;
    const double INF = std::numeric_limits<double>::infinity();
    bi.bxmin.assign(nb, INF);
    bi.bxmax.assign(nb, -INF);
    bi.bymin.assign(nb, INF);
    bi.bymax.assign(nb, -INF);
    bi.bvmax.assign(nb, -INF);
    bi.start.assign(nb + 1, 0);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            if (grid[(int64_t)y * W + x] >= 0) bi.start[(y / B) * bi.nbx + (x / B) + 1]++;
    for (int b = 0; b < nb; ++b) bi.start[b + 1] += bi.start[b];
    bi.cells.resize(bi.start[nb]);
    std::vector<int32_t> fill(bi.start.begin(), bi.start.end() - 1);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const int32_t id = grid[(int64_t)y * W + x];
            if (id < 0) continue;
            const int b = (y / B) * bi.nbx + (x / B);
            bi.cells[fill[b]++] = id;
            const double cx = C[2 * (int64_t)id], cy = C[2 * (int64_t)id + 1];
            bi.bxmin[b] = std::min(bi.bxmin[b], cx);
            bi.bxmax[b] = std::max(bi.bxmax[b], cx);
            bi.bymin[b] = std::min(bi.bymin[b], cy);
            bi.bymax[b] = std::max(bi.bymax[b], cy);
            bi.bvmax[b] = std::max(bi.bvmax[b], v[id]);
        }
    return bi;
}

// For each point: u[i] = min_j (c_ij - v_j) and argmin j; optionally collect up to
// max_viol cells j with c_ij - v_j < (c_i,own - v_own) - eps (the most violated first).
struct PointDual {
    double u;
    int32_t arg;
};

inline void point_min_reduced(const BlockIndex& bi, const double* C, const double* v, double px, double py,
                              double best_init, int32_t arg_init, PointDual& out, double thresh, int max_viol,
                              std::vector<std::pair<double, int32_t>>& viol, std::vector<double>& lbs) {
    const int nb = bi.nbx * bi.nby;
    lbs.resize(nb);
    double best = best_init;
    int32_t arg = arg_init;
    // lower bound for each block
    for (int b = 0; b < nb; ++b) {
        if (bi.start[b] == bi.start[b + 1]) {
            lbs[b] = std::numeric_limits<double>::infinity();
            continue;
        }
        const double dx = px < bi.bxmin[b] ? bi.bxmin[b] - px : (px > bi.bxmax[b] ? px - bi.bxmax[b] : 0.0);
        const double dy = py < bi.bymin[b] ? bi.bymin[b] - py : (py > bi.bymax[b] ? py - bi.bymax[b] : 0.0);
        lbs[b] = dx * dx + dy * dy - bi.bvmax[b];
    }
    // own block first for a good incumbent
    const double lim_v = max_viol > 0 ? thresh : -std::numeric_limits<double>::infinity();
    for (int pass = 0; pass < 2; ++pass) {
        for (int b = 0; b < nb; ++b) {
            const bool own = (px >= bi.bxmin[b] - 0.5 && px <= bi.bxmax[b] + 0.5 && py >= bi.bymin[b] - 0.5 &&
                              py <= bi.bymax[b] + 0.5);
            if ((pass == 0) != own) continue;
            const double cut = std::max(best, lim_v);
            if (lbs[b] >= cut) continue;
            for (int32_t t = bi.start[b]; t < bi.start[b + 1]; ++t) {
                const int32_t j = bi.cells[t];
                const double dx = px - C[2 * (int64_t)j], dy = py - C[2 * (int64_t)j + 1];
                const double r = dx * dx + dy * dy - v[j];
                if (r < best) {
                    best = r;
                    arg = j;
                }
                if (max_viol > 0 && r < thresh) viol.push_back({r, j});
            }
        }
    }
    out.u = best;
    out.arg = arg;
    if ((int)viol.size() > max_viol) {
        std::partial_sort(viol.begin(), viol.begin() + max_viol, viol.end());
        viol.resize(max_viol);
    }
}

// Single-source-from-everywhere Bellman-Ford (SPFA with small-label-first) on integer
// arc weights.  Returns false only if a negative cycle is verified: with the SLF
// heuristic the enqueue count is not a valid detector by itself, so when a node has
// been enqueued more than n times we walk its parent pointers and look for a cycle of
// negative total weight; if none is found the count is reset and the search continues.
inline bool spfa_potentials(int64_t n, const std::vector<int64_t>& start, const std::vector<int32_t>& head,
                            const std::vector<int64_t>& w, std::vector<int64_t>& d) {
    d.assign(n, 0);
    std::vector<char> inq(n, 1);
    std::vector<int64_t> cnt(n, 0);
    std::vector<int32_t> parent(n, -1);
    std::vector<int64_t> pw(n, 0);  // weight of the arc parent -> node
    std::vector<int64_t> mark(n, -1);
    int64_t stamp = 0;
    std::deque<int32_t> q;
    for (int64_t i = 0; i < n; ++i) q.push_back((int32_t)i);
    auto negative_cycle_through = [&](int32_t v0) -> bool {
        // walk parents from v0; a repeated node closes a cycle
        ++stamp;
        int32_t x = v0;
        for (int64_t steps = 0; steps <= n && x >= 0; ++steps) {
            if (mark[x] == stamp) {
                int64_t tot = 0;
                int32_t y = x;
                do {
                    tot += pw[y];
                    y = parent[y];
                } while (y != x && y >= 0);
                return y == x && tot < 0;
            }
            mark[x] = stamp;
            x = parent[x];
        }
        return false;
    };
    while (!q.empty()) {
        const int32_t a = q.front();
        q.pop_front();
        inq[a] = 0;
        const int64_t da = d[a];
        for (int64_t e = start[a]; e < start[a + 1]; ++e) {
            const int32_t b = head[e];
            const int64_t nd = da + w[e];
            if (nd < d[b]) {
                d[b] = nd;
                parent[b] = a;
                pw[b] = w[e];
                if (!inq[b]) {
                    if (++cnt[b] > n) {
                        if (negative_cycle_through(b)) return false;
                        cnt[b] = 0;
                    }
                    inq[b] = 1;
                    if (!q.empty() && nd < d[q.front()])
                        q.push_front(b);
                    else
                        q.push_back(b);
                }
            }
        }
    }
    return true;
}

}  // namespace shufflesnap
