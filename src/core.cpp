#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

constexpr int kMaxWindowCells = 36;
constexpr double kInfinity = std::numeric_limits<double>::infinity();

struct WindowTargets {
    int len = 0;
    std::array<int, kMaxWindowCells> target_ids{};
};

struct AssignmentResult {
    std::vector<std::int64_t> row_ind;
    std::vector<std::int64_t> col_ind;
    double total_cost = 0.0;
};

struct CleanupResult {
    std::vector<std::int64_t> assignment;
    std::int64_t rounds_completed = 0;
    double elapsed_s = 0.0;
    double final_cost = 0.0;
    bool converged = false;
    std::vector<double> round_elapsed_s;
    std::vector<double> round_costs;
    std::vector<std::int64_t> round_strides_row;
    std::vector<std::int64_t> round_strides_col;
};

struct WindowSolveResult {
    int len = 0;
    bool solved = false;
    std::array<int, kMaxWindowCells> point_ids{};
    std::array<int, kMaxWindowCells> assigned_target_ids{};
};

double squared_distance(double ax, double ay, double bx, double by) {
    const double dx = ax - bx;
    const double dy = ay - by;
    return (dx * dx) + (dy * dy);
}

int resolve_thread_count(int requested, std::size_t max_tasks) {
    if (requested < 0) {
        throw std::runtime_error("num_threads must be non-negative");
    }
    int thread_count = requested;
    if (thread_count == 0) {
        const unsigned int detected = std::thread::hardware_concurrency();
        thread_count = detected == 0 ? 1 : static_cast<int>(detected);
    }
    thread_count = std::max(1, thread_count);
    if (max_tasks > 0) {
        thread_count = std::min<int>(thread_count, static_cast<int>(max_tasks));
    }
    return thread_count;
}

AssignmentResult solve_square_jv_dense(const double* cost, std::size_t n) {
    if (n == 0) {
        return AssignmentResult{};
    }

    std::vector<double> u(n, 0.0);
    std::vector<double> v(n, 0.0);
    std::vector<double> shortest(n, kInfinity);
    std::vector<std::int64_t> path(n, -1);
    std::vector<std::int64_t> col4row(n, -1);
    std::vector<std::int64_t> row4col(n, -1);
    std::vector<std::size_t> remaining(n, 0);
    std::vector<unsigned char> sr(n, 0);
    std::vector<unsigned char> sc(n, 0);

    for (std::size_t cur_row = 0; cur_row < n; ++cur_row) {
        double min_val = 0.0;
        std::size_t i = cur_row;
        std::size_t num_remaining = n;

        for (std::size_t it = 0; it < n; ++it) {
            remaining[it] = n - it - 1;
            shortest[it] = kInfinity;
            path[it] = -1;
            sr[it] = 0;
            sc[it] = 0;
        }

        std::size_t sink = n;
        while (sink == n) {
            std::size_t index = n;
            double lowest = kInfinity;
            sr[i] = 1;

            const std::size_t row_offset = i * n;
            for (std::size_t it = 0; it < num_remaining; ++it) {
                const std::size_t j = remaining[it];
                const double reduced_cost = min_val + cost[row_offset + j] - u[i] - v[j];
                if (reduced_cost < shortest[j]) {
                    path[j] = static_cast<std::int64_t>(i);
                    shortest[j] = reduced_cost;
                }
                if (shortest[j] < lowest || (shortest[j] == lowest && row4col[j] == -1)) {
                    lowest = shortest[j];
                    index = it;
                }
            }

            min_val = lowest;
            if (!std::isfinite(min_val)) {
                throw std::runtime_error("infeasible assignment");
            }

            const std::size_t j = remaining[index];
            if (row4col[j] == -1) {
                sink = j;
            } else {
                i = static_cast<std::size_t>(row4col[j]);
            }
            sc[j] = 1;
            --num_remaining;
            remaining[index] = remaining[num_remaining];
        }

        u[cur_row] += min_val;
        for (std::size_t row = 0; row < n; ++row) {
            if (sr[row] && row != cur_row) {
                const std::size_t col = static_cast<std::size_t>(col4row[row]);
                u[row] += min_val - shortest[col];
            }
        }
        for (std::size_t col = 0; col < n; ++col) {
            if (sc[col]) {
                v[col] -= min_val - shortest[col];
            }
        }

        std::size_t j = sink;
        while (true) {
            const auto row = static_cast<std::size_t>(path[j]);
            row4col[j] = static_cast<std::int64_t>(row);
            const auto previous_j = col4row[row];
            col4row[row] = static_cast<std::int64_t>(j);
            if (row == cur_row) {
                break;
            }
            j = static_cast<std::size_t>(previous_j);
        }
    }

    AssignmentResult result;
    result.row_ind.resize(n);
    result.col_ind.resize(n);
    for (std::size_t row = 0; row < n; ++row) {
        result.row_ind[row] = static_cast<std::int64_t>(row);
        result.col_ind[row] = col4row[row];
        result.total_cost += cost[(row * n) + static_cast<std::size_t>(col4row[row])];
    }
    return result;
}

// Rectangular Jonker-Volgenant for small problems: assigns each of n_rows
// points to a distinct column out of n_cols >= n_rows. cost is row-major
// n_rows x n_cols. Returns false only if the search encounters a non-finite
// bound, which cannot happen for finite costs.
bool solve_rect_jv_small(int n_rows, int n_cols, const double* cost, int* col4row_out) {
    std::array<double, kMaxWindowCells> u{};
    std::array<double, kMaxWindowCells> v{};
    std::array<double, kMaxWindowCells> shortest{};
    std::array<int, kMaxWindowCells> path{};
    std::array<int, kMaxWindowCells> col4row{};
    std::array<int, kMaxWindowCells> row4col{};
    std::array<int, kMaxWindowCells> remaining{};
    std::array<unsigned char, kMaxWindowCells> sr{};
    std::array<unsigned char, kMaxWindowCells> sc{};

    col4row.fill(-1);
    row4col.fill(-1);

    for (int cur_row = 0; cur_row < n_rows; ++cur_row) {
        double min_val = 0.0;
        int i = cur_row;
        int num_remaining = n_cols;
        for (int it = 0; it < n_cols; ++it) {
            remaining[it] = n_cols - it - 1;
            shortest[it] = kInfinity;
            path[it] = -1;
            sc[it] = 0;
        }
        for (int it = 0; it < n_rows; ++it) {
            sr[it] = 0;
        }

        int sink = -1;
        while (sink < 0) {
            int index = -1;
            double lowest = kInfinity;
            sr[i] = 1;

            const int row_offset = i * n_cols;
            for (int it = 0; it < num_remaining; ++it) {
                const int j = remaining[it];
                const double reduced_cost = min_val + cost[row_offset + j] - u[i] - v[j];
                if (reduced_cost < shortest[j]) {
                    path[j] = i;
                    shortest[j] = reduced_cost;
                }
                if (shortest[j] < lowest || (shortest[j] == lowest && row4col[j] == -1)) {
                    lowest = shortest[j];
                    index = it;
                }
            }

            min_val = lowest;
            if (!std::isfinite(min_val)) {
                return false;
            }

            const int j = remaining[index];
            if (row4col[j] == -1) {
                sink = j;
            } else {
                i = row4col[j];
            }
            sc[j] = 1;
            --num_remaining;
            remaining[index] = remaining[num_remaining];
        }

        u[cur_row] += min_val;
        for (int row = 0; row < n_rows; ++row) {
            if (sr[row] && row != cur_row) {
                const int col = col4row[row];
                u[row] += min_val - shortest[col];
            }
        }
        for (int col = 0; col < n_cols; ++col) {
            if (sc[col]) {
                v[col] -= min_val - shortest[col];
            }
        }

        int j = sink;
        while (true) {
            const int row = path[j];
            row4col[j] = row;
            const int previous_j = col4row[row];
            col4row[row] = j;
            if (row == cur_row) {
                break;
            }
            j = previous_j;
        }
    }

    for (int row = 0; row < n_rows; ++row) {
        col4row_out[row] = col4row[row];
    }
    return true;
}

void build_target_grid(
    int rows,
    int cols,
    double margin,
    std::vector<double>& target_x,
    std::vector<double>& target_y
) {
    const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    target_x.assign(n, 0.0);
    target_y.assign(n, 0.0);

    std::vector<double> xs(cols, 0.5);
    std::vector<double> ys(rows, 0.5);
    if (cols > 1) {
        const double step = (1.0 - (2.0 * margin)) / static_cast<double>(cols - 1);
        for (int col = 0; col < cols; ++col) {
            xs[col] = margin + (step * static_cast<double>(col));
        }
    }
    if (rows > 1) {
        const double step = (1.0 - (2.0 * margin)) / static_cast<double>(rows - 1);
        for (int row = 0; row < rows; ++row) {
            ys[row] = margin + (step * static_cast<double>(row));
        }
    }

    for (int row = 0; row < rows; ++row) {
        const int row_offset = row * cols;
        for (int col = 0; col < cols; ++col) {
            const int target_id = row_offset + col;
            target_x[target_id] = xs[col];
            target_y[target_id] = ys[row];
        }
    }
}

// Windows for one phase at one (row, col) stride. The grid cells split into
// stride_r * stride_c interleaved cosets; coset (a, b) holds the cells
// (a + i * stride_r, b + j * stride_c) and forms a subgrid that is tiled with
// window_rows x window_cols windows offset by (row_phase, col_phase). All
// returned windows are pairwise disjoint, so they can be solved in parallel.
std::vector<WindowTargets> build_phase_windows(
    int rows,
    int cols,
    int window_rows,
    int window_cols,
    int row_phase,
    int col_phase,
    int stride_r,
    int stride_c
) {
    std::vector<WindowTargets> windows;
    for (int a = 0; a < stride_r; ++a) {
        const int sub_rows = (rows - a + stride_r - 1) / stride_r;
        if (sub_rows <= 0) {
            continue;
        }
        for (int b = 0; b < stride_c; ++b) {
            const int sub_cols = (cols - b + stride_c - 1) / stride_c;
            if (sub_cols <= 0) {
                continue;
            }
            int row_start = std::min(row_phase, sub_rows - 1);
            while (true) {
                const int row_end = std::min(row_start + window_rows, sub_rows);
                int col_start = std::min(col_phase, sub_cols - 1);
                while (true) {
                    const int col_end = std::min(col_start + window_cols, sub_cols);
                    WindowTargets window;
                    for (int row = row_start; row < row_end; ++row) {
                        const int grid_row = a + (row * stride_r);
                        const int row_offset = grid_row * cols;
                        for (int col = col_start; col < col_end; ++col) {
                            window.target_ids[window.len++] = row_offset + b + (col * stride_c);
                        }
                    }
                    if (window.len > 1) {
                        windows.push_back(window);
                    }
                    if (col_end == sub_cols) {
                        break;
                    }
                    col_start += window_cols;
                }
                if (row_end == sub_rows) {
                    break;
                }
                row_start += window_rows;
            }
        }
    }
    return windows;
}

CleanupResult run_cleanup(
    const double* points,
    std::size_t n_points,
    const std::int64_t* initial_assignment,
    int rows,
    int cols,
    double budget_seconds,
    int window_size,
    double margin,
    int fixed_suffix_count,
    int num_threads,
    const std::vector<std::int64_t>& stride_schedule,
    bool trace_rounds,
    const std::vector<std::int64_t>& blocked_cells
) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("rows and cols must be positive");
    }
    const std::size_t n_cells = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    if (n_cells > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("rows * cols must fit in a 32-bit signed integer");
    }
    if (n_points > n_cells) {
        throw std::runtime_error("number of points must not exceed rows * cols");
    }
    if (window_size <= 0 || window_size > 6) {
        throw std::runtime_error("window_size must be in the range [1, 6] for the current native kernel");
    }
    if (fixed_suffix_count < 0 || static_cast<std::size_t>(fixed_suffix_count) > n_cells) {
        throw std::runtime_error("fixed_suffix_count must be in [0, rows * cols]");
    }
    if (num_threads < 0) {
        throw std::runtime_error("num_threads must be non-negative");
    }
    if (stride_schedule.empty() || stride_schedule.size() % 2 != 0) {
        throw std::runtime_error("stride schedule must be a non-empty flat list of (row, col) pairs");
    }
    for (const auto stride : stride_schedule) {
        if (stride < 1) {
            throw std::runtime_error("strides must be positive");
        }
    }

    std::vector<double> ax(n_points, 0.0);
    std::vector<double> ay(n_points, 0.0);
    for (std::size_t i = 0; i < n_points; ++i) {
        ax[i] = points[(2 * i) + 0];
        ay[i] = points[(2 * i) + 1];
    }

    std::vector<unsigned char> usable;
    if (!blocked_cells.empty()) {
        usable.assign(n_cells, 1);
        for (const auto blocked : blocked_cells) {
            if (blocked < 0 || blocked >= static_cast<std::int64_t>(n_cells)) {
                throw std::runtime_error("cell mask contains out-of-range cell ids");
            }
            usable[static_cast<std::size_t>(blocked)] = 0;
        }
        if (n_points > n_cells - blocked_cells.size()) {
            throw std::runtime_error("cell mask leaves fewer usable cells than points");
        }
    }

    std::vector<int> assignment(n_points, -1);
    std::vector<int> owner(n_cells, -1);
    for (std::size_t point_id = 0; point_id < n_points; ++point_id) {
        const auto target_id = static_cast<std::int64_t>(initial_assignment[point_id]);
        if (target_id < 0 || target_id >= static_cast<std::int64_t>(n_cells)) {
            throw std::runtime_error("initial_assignment contains out-of-range target ids");
        }
        if (!usable.empty() && usable[static_cast<std::size_t>(target_id)] == 0) {
            throw std::runtime_error("initial_assignment assigns a point to a masked-out cell");
        }
        if (owner[static_cast<std::size_t>(target_id)] != -1) {
            throw std::runtime_error("initial_assignment assigns two points to the same target");
        }
        assignment[point_id] = static_cast<int>(target_id);
        owner[static_cast<std::size_t>(target_id)] = static_cast<int>(point_id);
    }

    std::vector<double> target_x;
    std::vector<double> target_y;
    build_target_grid(rows, cols, margin, target_x, target_y);
    const int fixed_start = static_cast<int>(n_cells) - fixed_suffix_count;

    const int half = std::max(1, window_size / 2);
    const std::array<std::array<int, 2>, 4> phase_offsets = {{{0, 0}, {0, half}, {half, 0}, {half, half}}};
    const std::size_t schedule_len = stride_schedule.size() / 2;

    const auto assignment_total_cost = [&]() {
        double total = 0.0;
        for (std::size_t point_id = 0; point_id < n_points; ++point_id) {
            const auto target_id = static_cast<std::size_t>(assignment[point_id]);
            total += squared_distance(ax[point_id], ay[point_id], target_x[target_id], target_y[target_id]);
        }
        return total;
    };

    auto start = std::chrono::steady_clock::now();
    CleanupResult result;
    std::int64_t rounds = 0;
    bool converged = false;

    for (;;) {
        const std::size_t schedule_index = std::min(static_cast<std::size_t>(rounds), schedule_len - 1);
        const int stride_r = static_cast<int>(stride_schedule[2 * schedule_index]);
        const int stride_c = static_cast<int>(stride_schedule[(2 * schedule_index) + 1]);
        std::int64_t round_changes = 0;

        for (const auto& offsets : phase_offsets) {
            const std::vector<WindowTargets> phase = build_phase_windows(
                rows, cols, window_size, window_size, offsets[0], offsets[1], stride_r, stride_c
            );
            if (phase.empty()) {
                continue;
            }
            std::vector<WindowSolveResult> results(phase.size());
            std::atomic<int> phase_failed{0};
            std::mutex worker_exception_mutex;
            std::exception_ptr worker_exception;
            const int thread_count = resolve_thread_count(num_threads, phase.size());
            auto solve_window_range = [&](std::size_t begin, std::size_t end) {
                try {
                    for (std::size_t window_idx = begin; window_idx < end; ++window_idx) {
                        if (phase_failed.load(std::memory_order_relaxed) != 0) {
                            return;
                        }
                        const auto& window = phase[window_idx];
                        WindowSolveResult local;
                        std::array<int, kMaxWindowCells> active_targets{};
                        int cell_count = 0;
                        for (int i = 0; i < window.len; ++i) {
                            const int target_id = window.target_ids[i];
                            if (target_id >= fixed_start) {
                                continue;
                            }
                            if (!usable.empty() && usable[static_cast<std::size_t>(target_id)] == 0) {
                                continue;
                            }
                            active_targets[cell_count++] = target_id;
                        }
                        if (cell_count <= 1) {
                            results[window_idx] = local;
                            continue;
                        }

                        int point_count = 0;
                        std::array<int, kMaxWindowCells> window_points{};
                        std::array<int, kMaxWindowCells> incumbent_cols{};
                        for (int i = 0; i < cell_count; ++i) {
                            const int point_id = owner[static_cast<std::size_t>(active_targets[i])];
                            if (point_id >= 0) {
                                incumbent_cols[point_count] = i;
                                window_points[point_count++] = point_id;
                            }
                        }
                        if (point_count == 0) {
                            results[window_idx] = local;
                            continue;
                        }

                        std::array<double, kMaxWindowCells * kMaxWindowCells> cost{};
                        std::array<int, kMaxWindowCells> col4row{};
                        for (int i = 0; i < point_count; ++i) {
                            const int point_id = window_points[i];
                            const double px = ax[static_cast<std::size_t>(point_id)];
                            const double py = ay[static_cast<std::size_t>(point_id)];
                            const int row_offset = i * cell_count;
                            for (int j = 0; j < cell_count; ++j) {
                                const int target_j = active_targets[j];
                                cost[static_cast<std::size_t>(row_offset + j)] = squared_distance(
                                    px,
                                    py,
                                    target_x[static_cast<std::size_t>(target_j)],
                                    target_y[static_cast<std::size_t>(target_j)]
                                );
                            }
                        }

                        if (!solve_rect_jv_small(point_count, cell_count, cost.data(), col4row.data())) {
                            phase_failed.store(1, std::memory_order_relaxed);
                            return;
                        }

                        // Apply only solutions that improve the window cost by
                        // more than the summation rounding error (a 36-term sum
                        // is exact to ~1e-14 relative). This keeps the true
                        // global cost strictly decreasing across applied windows,
                        // which guarantees termination and prevents endless
                        // flips between tied optima (e.g. duplicate points).
                        double incumbent_cost = 0.0;
                        double new_cost = 0.0;
                        for (int i = 0; i < point_count; ++i) {
                            const int row_offset = i * cell_count;
                            incumbent_cost += cost[static_cast<std::size_t>(row_offset + incumbent_cols[i])];
                            new_cost += cost[static_cast<std::size_t>(row_offset + col4row[i])];
                        }
                        if (incumbent_cost - new_cost <= 1e-12 * incumbent_cost) {
                            results[window_idx] = local;
                            continue;
                        }

                        local.solved = true;
                        local.len = point_count;
                        for (int i = 0; i < point_count; ++i) {
                            local.point_ids[static_cast<std::size_t>(i)] = window_points[i];
                            local.assigned_target_ids[static_cast<std::size_t>(i)] =
                                active_targets[static_cast<std::size_t>(col4row[static_cast<std::size_t>(i)])];
                        }
                        results[window_idx] = local;
                    }
                } catch (...) {
                    phase_failed.store(1, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lock(worker_exception_mutex);
                    if (!worker_exception) {
                        worker_exception = std::current_exception();
                    }
                }
            };

            if (thread_count == 1) {
                solve_window_range(0, phase.size());
            } else {
                std::vector<std::thread> workers;
                workers.reserve(static_cast<std::size_t>(thread_count));
                const std::size_t base_chunk = phase.size() / static_cast<std::size_t>(thread_count);
                const std::size_t remainder = phase.size() % static_cast<std::size_t>(thread_count);
                std::size_t begin = 0;
                for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
                    const std::size_t extra = static_cast<std::size_t>(thread_idx) < remainder ? 1 : 0;
                    const std::size_t end = begin + base_chunk + extra;
                    workers.emplace_back(solve_window_range, begin, end);
                    begin = end;
                }
                for (auto& worker : workers) {
                    worker.join();
                }
            }

            if (worker_exception) {
                std::rethrow_exception(worker_exception);
            }

            if (phase_failed.load(std::memory_order_relaxed) != 0) {
                throw std::runtime_error("window LAP failed");
            }

            for (const auto& solved : results) {
                if (!solved.solved) {
                    continue;
                }
                for (int i = 0; i < solved.len; ++i) {
                    const int point_id = solved.point_ids[static_cast<std::size_t>(i)];
                    const int target_id = solved.assigned_target_ids[static_cast<std::size_t>(i)];
                    const int previous_target = assignment[static_cast<std::size_t>(point_id)];
                    if (previous_target == target_id) {
                        continue;
                    }
                    if (owner[static_cast<std::size_t>(previous_target)] == point_id) {
                        owner[static_cast<std::size_t>(previous_target)] = -1;
                    }
                    assignment[static_cast<std::size_t>(point_id)] = target_id;
                    owner[static_cast<std::size_t>(target_id)] = point_id;
                    ++round_changes;
                }
            }
        }

        ++rounds;
        const auto now = std::chrono::steady_clock::now();
        const double elapsed_s = std::chrono::duration<double>(now - start).count();
        const bool finest = (stride_r == 1) && (stride_c == 1);
        const bool schedule_done = static_cast<std::size_t>(rounds) >= schedule_len;

        if (trace_rounds) {
            result.round_elapsed_s.push_back(elapsed_s);
            result.round_costs.push_back(assignment_total_cost());
            result.round_strides_row.push_back(static_cast<std::int64_t>(stride_r));
            result.round_strides_col.push_back(static_cast<std::int64_t>(stride_c));
        }

        if (finest && schedule_done && round_changes == 0) {
            converged = true;
        }
        if (converged || elapsed_s >= budget_seconds) {
            result.assignment.assign(assignment.begin(), assignment.end());
            result.rounds_completed = rounds;
            result.elapsed_s = elapsed_s;
            result.final_cost = assignment_total_cost();
            result.converged = converged;
            return result;
        }
    }
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "Native dense LAP and multiscale window cleanup kernels";

    m.def(
        "_linear_sum_assignment",
        [](nb::ndarray<const double, nb::numpy, nb::c_contig> cost_matrix) {
            if (cost_matrix.ndim() != 2) {
                throw std::runtime_error("cost_matrix must be 2D");
            }
            const std::size_t rows = cost_matrix.shape(0);
            const std::size_t cols = cost_matrix.shape(1);
            if (rows != cols) {
                throw std::runtime_error("cost_matrix must be square");
            }
            const auto* cost = static_cast<const double*>(cost_matrix.data());
            AssignmentResult result;
            {
                nb::gil_scoped_release release;
                result = solve_square_jv_dense(cost, rows);
            }
            return nb::make_tuple(result.row_ind, result.col_ind, result.total_cost);
        },
        "cost_matrix"_a,
        "Solve a dense square LAP with a native JV backend."
    );

    m.def(
        "_window_cleanup",
        [](nb::ndarray<const double, nb::numpy, nb::c_contig> points,
           nb::ndarray<const std::int64_t, nb::numpy, nb::c_contig> initial_assignment,
           int rows,
           int cols,
           double budget_seconds,
           int window_size,
           double margin,
           int fixed_suffix_count,
           int num_threads,
           std::vector<std::int64_t> stride_schedule,
           bool trace_rounds,
           std::vector<std::int64_t> blocked_cells) {
            if (points.ndim() != 2 || points.shape(1) != 2) {
                throw std::runtime_error("points must have shape (n, 2)");
            }
            if (initial_assignment.ndim() != 1) {
                throw std::runtime_error("initial_assignment must be 1D");
            }
            const std::size_t n_points = points.shape(0);
            if (initial_assignment.shape(0) != n_points) {
                throw std::runtime_error("initial_assignment must have length n");
            }
            const auto* point_ptr = static_cast<const double*>(points.data());
            const auto* assignment_ptr = static_cast<const std::int64_t*>(initial_assignment.data());
            CleanupResult result;
            {
                nb::gil_scoped_release release;
                result = run_cleanup(
                    point_ptr,
                    n_points,
                    assignment_ptr,
                    rows,
                    cols,
                    budget_seconds,
                    window_size,
                    margin,
                    fixed_suffix_count,
                    num_threads,
                    stride_schedule,
                    trace_rounds,
                    blocked_cells
                );
            }
            nb::dict out;
            out["assignment"] = nb::cast(result.assignment);
            out["rounds_completed"] = nb::int_(result.rounds_completed);
            out["elapsed_s"] = nb::float_(result.elapsed_s);
            out["final_cost"] = nb::float_(result.final_cost);
            out["converged"] = nb::bool_(result.converged);
            if (trace_rounds) {
                out["round_elapsed_s"] = nb::cast(result.round_elapsed_s);
                out["round_costs"] = nb::cast(result.round_costs);
                out["round_strides"] = nb::cast(result.round_strides_row);
                out["round_strides_col"] = nb::cast(result.round_strides_col);
            }
            return out;
        },
        "points"_a,
        "initial_assignment"_a,
        "rows"_a,
        "cols"_a,
        "budget_seconds"_a,
        "window_size"_a = 6,
        "margin"_a = 0.03,
        "fixed_suffix_count"_a = 0,
        "num_threads"_a = 0,
        "stride_schedule"_a,
        "trace_rounds"_a = false,
        "blocked_cells"_a = std::vector<std::int64_t>{},
        "Run native multiscale window cleanup from an initial assignment."
    );
}
