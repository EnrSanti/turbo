// Copyright 2021 Pierre Talbot

#ifndef TURBO_STATISTICS_HPP
#define TURBO_STATISTICS_HPP

#include <chrono>
#include <algorithm>
#include "battery/utility.hpp"
#include "battery/allocator.hpp"
#include "lala/logic/ast.hpp"

enum class Timer {
  OVERALL,
  PREPROCESSING,
  SEARCH,
  // SPLIT,
  // PUSH,
  // POP,
  FIXPOINT,
  TRANSFER_CPU2GPU,
  TRANSFER_GPU2CPU,
  SELECT_FP_FUNCTIONS,
  WAIT_CPU,
  DIVE,
  LATEST_BEST_OBJ_FOUND,
  FIRST_BLOCK_IDLE,
  BACKTRACK_ENR, //ENR STATS: time spent restoring the store from the subproblem root on backtrack.
  NUM_TIMERS
};

template <class Allocator = battery::standard_allocator>
struct TimingStatistics {
  using allocator_type = Allocator;
  template <class Alloc> friend struct TimingStatistics;
private:
  battery::vector<int64_t, allocator_type> timers_ns;

public:

  TimingStatistics(const TimingStatistics&) = default;
  CUDA TimingStatistics():
    timers_ns(static_cast<int>(Timer::NUM_TIMERS), 0)
  {}

  template <class Alloc>
  CUDA TimingStatistics(const TimingStatistics<Alloc>& other):
    timers_ns(other.timers_ns)
  {}

  CUDA int64_t& time_of(Timer timer) {
    return timers_ns[static_cast<int>(timer)];
  }

  CUDA int64_t time_of(Timer timer) const {
    return timers_ns[static_cast<int>(timer)];
  }

  template <class Alloc>
  CUDA void meet(const TimingStatistics<Alloc>& other) {
    for(int i = 0; i < timers_ns.size(); i++) {
      timers_ns[i] += other.timers_ns[i];
    }
  }

  CUDA int64_t time_ms_of(Timer timer) const {
    return timers_ns[static_cast<int>(timer)] / 1000 / 1000;
  }

#ifdef __CUDACC__
  __device__ cuda::std::chrono::system_clock::time_point start_timer_device() const {
    if(threadIdx.x == 0) {
      return cuda::std::chrono::system_clock::now();
    }
    else {
      return cuda::std::chrono::system_clock::time_point{};
    }
  }
#endif

  std::chrono::steady_clock::time_point start_timer_host() const {
    return std::chrono::steady_clock::now();
  }

#ifdef __CUDACC__
  __device__ cuda::std::chrono::system_clock::time_point stop_timer(Timer timer, cuda::std::chrono::system_clock::time_point start) {
    if(threadIdx.x == 0) {
      auto now = cuda::std::chrono::system_clock::now();
      time_of(timer) += cuda::std::chrono::duration_cast<cuda::std::chrono::nanoseconds>(
        now - start).count();
      return now;
    }
    return cuda::std::chrono::system_clock::time_point{};
  }

  __device__ void update_timer(Timer timer, cuda::std::chrono::system_clock::time_point start) {
    if(threadIdx.x == 0) {
      auto now = cuda::std::chrono::system_clock::now();
      time_of(timer) = cuda::std::chrono::duration_cast<cuda::std::chrono::nanoseconds>(
        now - start).count();
    }
  }
#endif

  std::chrono::steady_clock::time_point stop_timer(Timer timer, std::chrono::steady_clock::time_point start) {
    auto now = std::chrono::steady_clock::now();
    time_of(timer) += std::chrono::duration_cast<std::chrono::nanoseconds>(
      now - start).count();
    return now;
  }

  /** Replace the value of the timer with `now - start`. */
  void update_timer(Timer timer, std::chrono::steady_clock::time_point start) {
    auto now = std::chrono::steady_clock::now();
    time_of(timer) = std::chrono::duration_cast<std::chrono::nanoseconds>(
      now - start).count();
  }
};

template <class Allocator = battery::standard_allocator>
struct Statistics {
  template <class Alloc> friend struct Statistics;
  using allocator_type = Allocator;
  bool print_statistics; // from config.print_statistics
  size_t variables;
  size_t constraints;
  bool optimization;
  int num_blocks; // The real count of config.or_nodes on GPU.
  size_t nodes;
  size_t fails;
  size_t solutions;
  int depth_max;
  bool exhaustive;
  size_t eps_num_subproblems;
  size_t eps_solved_subproblems;
  size_t eps_skipped_subproblems;
  size_t num_blocks_done;
  size_t fixpoint_iterations;
  size_t num_deductions;
  int64_t cumulative_time_block;
  TimingStatistics<Allocator> timers;
  //ENR STATS -- begin
  // Nodes of the search tree counted once: a dive node shared by several subproblems is counted only by the
  // subproblem that visits it first (lowest index), so enr_tree_nodes = number of distinct nodes explored.
  size_t enr_tree_nodes;
  size_t enr_tree_fails;
  // The fields below are only filled by `reduce_blocks` from the grid data (not by `meet`).
  bool enr_restart_mode;                 // -restart > 0 (barebones).
  size_t enr_restarts;                   // number of load-balancing restarts.
  size_t enr_restart_items_sum;          // open nodes collected, over all restarts.
  size_t enr_restart_items_max;          // largest number of open nodes collected at one restart.
  size_t enr_restart_items_last;         // open nodes collected at the last restart.
  size_t enr_restart_subproblems_sum;    // subproblems (n * 2^s) handed out after the restarts.
  int enr_restart_s_max;                 // largest split depth s used after a restart.
  int enr_restart_truncated;             // blocks that had to truncate their open nodes (item quota exceeded).
  int64_t enr_restart_drain_ns_sum;      // time between the restart flag and the moment all blocks have stopped (S1).
  int64_t enr_restart_drain_ns_max;
  int64_t enr_restart_first_ns;          // kernel time of the first restart (-1 if none).
  int64_t enr_restart_last_ns;           // kernel time of the last restart (-1 if none).
  //ENR STATS -- end
  //ENR STATS -- begin
  static constexpr int ENR_HIST = 16; // bucket 0: <1us, bucket b>=1: [2^(b-1), 2^b) us, last bucket is open-ended.
  size_t enr_node_fp_hist[ENR_HIST];   // histogram of the fixpoint duration of single nodes.
  int64_t enr_node_fp_max_ns;           // slowest fixpoint of a single node.
  size_t enr_backtracks;                // number of restorations from the subproblem root.
  // Restarts after a failure inside the subproblem: restore (copy root_store + re-embed the path) + the next fixpoint.
  size_t enr_rs_count, enr_rs_rounds, enr_rs_path_sum, enr_rs_path_max;
  int64_t enr_rs_restore_ns, enr_rs_fp_ns, enr_rs_cost_max_ns;
  size_t enr_rs_hist[ENR_HIST];         // histogram of the cost (restore + fixpoint) of single restarts.
  // Nodes NOT reached by a restart (after a split, the dive, the subproblem root), for comparison.
  size_t enr_sp_count, enr_sp_rounds;
  int64_t enr_sp_fp_ns;
  // The fields below are only filled by `reduce_blocks` (cross-block load balance), not by `meet`.
  size_t enr_nodes_block_min, enr_nodes_block_max;
  size_t enr_subpb_block_min, enr_subpb_block_max;
  int64_t enr_life_block_min, enr_life_block_max; // lifetime (ns) of a block in the kernel.
  int64_t enr_idle_ns_sum;              // sum over blocks of (longest block lifetime - block lifetime).
  static constexpr int ENR_TL = 20;     // idle timeline: the kernel time is cut into ENR_TL equal slices.
  size_t enr_done_blocks;               // blocks that left the loop because no subproblem was left (not stopped).
  int64_t enr_first_done_ns, enr_last_done_ns; // lifetime of the first / last such block, -1 if none.
  double enr_idle_pct_slice[ENR_TL];    // % of the block slots that were empty during slice k (time-weighted).
  int enr_busy_end_slice[ENR_TL];       // number of blocks still in the loop at the end of slice k.
  // Phases of a block (exclusive, contiguous: they sum to the block lifetime). Charged with ENR_PH in barebones.
  enum EnrPhase : int {
    PH_SUBPB_SWITCH = 0, // G (take the next subproblem) + loop head + C (copy of the global root); counter = dives
    PH_DIVE_FP,          // whole propagate() of a dive node (fixpoint + leaf check), timed at the call site; counter = dive nodes
    PH_DIVE_BRANCH,      // split + commit to the branch of the path, during the dive
    PH_SEARCH_FP,        // whole propagate() of a node reached by going down (left branch); counter = such nodes
    PH_SEARCH_BRANCH,    // objective bound, root snapshot, split, decision, rope lookup (solve phase F)
    PH_BT_RESTORE,       // backtrack: copy root_store -> store; counter = backtracks
    PH_BT_REDESCEND,     // backtrack: re-apply the decisions of the path + take the right branch; counter = backtracks
    PH_BT_FP,            // whole propagate() of the node reached by the redescend; counter = such nodes
    ENR_PH_N
  };
  int64_t enr_ph_ns[ENR_PH_N];          // time of each phase (sum over blocks after reduce).
  size_t enr_ph_cnt[ENR_PH_N];          // occurrences of each phase (see the comments above).
  //ENR STATS -- end

  CUDA Statistics(size_t variables, size_t constraints, bool optimization, bool print_statistics):
    variables(variables), constraints(constraints), optimization(optimization),
    print_statistics(print_statistics), num_blocks(1),
    nodes(0), fails(0), solutions(0),
    depth_max(0), exhaustive(true),
    eps_solved_subproblems(0), eps_num_subproblems(1), eps_skipped_subproblems(0),
    num_blocks_done(0), fixpoint_iterations(0), num_deductions(0),
    cumulative_time_block(0), timers()
  {
    enr_restart_reset(); //ENR STATS
    enr_reset(); //ENR STATS
  }

  //ENR STATS
  CUDA void enr_restart_reset() {
    enr_tree_nodes = 0; enr_tree_fails = 0;
    enr_restart_mode = false; enr_restarts = 0;
    enr_restart_items_sum = 0; enr_restart_items_max = 0; enr_restart_items_last = 0; enr_restart_subproblems_sum = 0;
    enr_restart_s_max = 0; enr_restart_truncated = 0;
    enr_restart_drain_ns_sum = 0; enr_restart_drain_ns_max = 0; enr_restart_first_ns = -1; enr_restart_last_ns = -1;
  }

  //ENR STATS
  template <class Alloc>
  CUDA void enr_restart_copy(const Statistics<Alloc>& other) {
    enr_tree_nodes = other.enr_tree_nodes; enr_tree_fails = other.enr_tree_fails;
    enr_restart_mode = other.enr_restart_mode; enr_restarts = other.enr_restarts;
    enr_restart_items_sum = other.enr_restart_items_sum; enr_restart_items_max = other.enr_restart_items_max; enr_restart_items_last = other.enr_restart_items_last;
    enr_restart_subproblems_sum = other.enr_restart_subproblems_sum;
    enr_restart_s_max = other.enr_restart_s_max; enr_restart_truncated = other.enr_restart_truncated;
    enr_restart_drain_ns_sum = other.enr_restart_drain_ns_sum; enr_restart_drain_ns_max = other.enr_restart_drain_ns_max;
    enr_restart_first_ns = other.enr_restart_first_ns; enr_restart_last_ns = other.enr_restart_last_ns;
  }

  //ENR STATS
  CUDA void enr_reset() {
    for(int i = 0; i < ENR_HIST; ++i) { enr_node_fp_hist[i] = 0; }
    enr_node_fp_max_ns = 0; enr_backtracks = 0;
    enr_rs_count = 0; enr_rs_rounds = 0; enr_rs_path_sum = 0; enr_rs_path_max = 0;
    enr_rs_restore_ns = 0; enr_rs_fp_ns = 0; enr_rs_cost_max_ns = 0;
    for(int i = 0; i < ENR_HIST; ++i) { enr_rs_hist[i] = 0; }
    enr_sp_count = 0; enr_sp_rounds = 0; enr_sp_fp_ns = 0;
    enr_nodes_block_min = 0; enr_nodes_block_max = 0;
    enr_subpb_block_min = 0; enr_subpb_block_max = 0;
    enr_life_block_min = 0; enr_life_block_max = 0;
    enr_idle_ns_sum = 0;
    enr_done_blocks = 0; enr_first_done_ns = -1; enr_last_done_ns = -1;
    for(int i = 0; i < ENR_TL; ++i) { enr_idle_pct_slice[i] = 0; enr_busy_end_slice[i] = 0; }
    for(int i = 0; i < ENR_PH_N; ++i) { enr_ph_ns[i] = 0; enr_ph_cnt[i] = 0; } //ENR STATS
  }

  //ENR STATS
  template <class Alloc>
  CUDA void enr_copy(const Statistics<Alloc>& other) {
    for(int i = 0; i < ENR_HIST; ++i) { enr_node_fp_hist[i] = other.enr_node_fp_hist[i]; }
    enr_node_fp_max_ns = other.enr_node_fp_max_ns; enr_backtracks = other.enr_backtracks;
    enr_rs_count = other.enr_rs_count; enr_rs_rounds = other.enr_rs_rounds; enr_rs_path_sum = other.enr_rs_path_sum; enr_rs_path_max = other.enr_rs_path_max;
    enr_rs_restore_ns = other.enr_rs_restore_ns; enr_rs_fp_ns = other.enr_rs_fp_ns; enr_rs_cost_max_ns = other.enr_rs_cost_max_ns;
    for(int i = 0; i < ENR_HIST; ++i) { enr_rs_hist[i] = other.enr_rs_hist[i]; }
    enr_sp_count = other.enr_sp_count; enr_sp_rounds = other.enr_sp_rounds; enr_sp_fp_ns = other.enr_sp_fp_ns;
    enr_nodes_block_min = other.enr_nodes_block_min; enr_nodes_block_max = other.enr_nodes_block_max;
    enr_subpb_block_min = other.enr_subpb_block_min; enr_subpb_block_max = other.enr_subpb_block_max;
    enr_life_block_min = other.enr_life_block_min; enr_life_block_max = other.enr_life_block_max;
    enr_idle_ns_sum = other.enr_idle_ns_sum;
    enr_done_blocks = other.enr_done_blocks; enr_first_done_ns = other.enr_first_done_ns; enr_last_done_ns = other.enr_last_done_ns;
    for(int i = 0; i < ENR_TL; ++i) { enr_idle_pct_slice[i] = other.enr_idle_pct_slice[i]; enr_busy_end_slice[i] = other.enr_busy_end_slice[i]; }
    for(int i = 0; i < ENR_PH_N; ++i) { enr_ph_ns[i] = other.enr_ph_ns[i]; enr_ph_cnt[i] = other.enr_ph_cnt[i]; } //ENR STATS
  }

  //ENR STATS: record the fixpoint duration of one node.
  CUDA static int enr_bucket(int64_t ns) {
    int64_t us = ns / 1000;
    int b = 0;
    while(us > 0 && b < ENR_HIST - 1) { us >>= 1; ++b; }
    return b;
  }

  CUDA void enr_record_node_fp(int64_t ns) {
    enr_node_fp_hist[enr_bucket(ns)]++;
    enr_node_fp_max_ns = enr_node_fp_max_ns < ns ? ns : enr_node_fp_max_ns;
  }

  //ENR STATS: one backtrack = restore + redescend (restore_ns) + the fixpoint of the node it leads to (fp_ns).
  CUDA void enr_record_backtrack(int64_t restore_ns, int64_t fp_ns, int rounds, int path_len) {
    int64_t cost = restore_ns + fp_ns;
    enr_rs_count++; enr_rs_rounds += rounds;
    enr_rs_path_sum += path_len; enr_rs_path_max = enr_rs_path_max < (size_t)path_len ? (size_t)path_len : enr_rs_path_max;
    enr_rs_restore_ns += restore_ns; enr_rs_fp_ns += fp_ns;
    enr_rs_cost_max_ns = enr_rs_cost_max_ns < cost ? cost : enr_rs_cost_max_ns;
    enr_rs_hist[enr_bucket(cost)]++;
  }

  //ENR STATS: a node not reached by a backtrack.
  CUDA void enr_record_other_node(int64_t fp_ns, int rounds) {
    enr_sp_count++; enr_sp_rounds += rounds; enr_sp_fp_ns += fp_ns;
  }

  CUDA Statistics(): Statistics(0,0,false,false) {}
  Statistics(const Statistics&) = default;
  Statistics(Statistics&&) = default;

  template <class Alloc>
  CUDA Statistics(const Statistics<Alloc>& other):
    variables(other.variables), constraints(other.constraints), optimization(other.optimization),
    print_statistics(other.print_statistics), num_blocks(other.num_blocks),
    nodes(other.nodes), fails(other.fails), solutions(other.solutions),
    depth_max(other.depth_max), exhaustive(other.exhaustive),
    eps_solved_subproblems(other.eps_solved_subproblems), eps_num_subproblems(other.eps_num_subproblems),
    eps_skipped_subproblems(other.eps_skipped_subproblems), num_blocks_done(other.num_blocks_done),
    fixpoint_iterations(other.fixpoint_iterations), num_deductions(other.num_deductions),
    cumulative_time_block(other.cumulative_time_block), timers(other.timers)
  {
    enr_restart_copy(other); //ENR STATS
    enr_copy(other); //ENR STATS
  }

  template <class Alloc>
  CUDA void meet(const Statistics<Alloc>& other) {
    nodes += other.nodes;
    fails += other.fails;
    solutions += other.solutions;
    depth_max = battery::max(depth_max, other.depth_max);
    exhaustive = exhaustive && other.exhaustive;
    eps_solved_subproblems += other.eps_solved_subproblems;
    eps_skipped_subproblems += other.eps_skipped_subproblems;
    num_blocks_done += other.num_blocks_done;
    fixpoint_iterations += other.fixpoint_iterations;
    num_deductions += other.num_deductions;
    cumulative_time_block += other.cumulative_time_block;
    timers.meet(other.timers);
    enr_tree_nodes += other.enr_tree_nodes; enr_tree_fails += other.enr_tree_fails; //ENR STATS
    //ENR STATS
    for(int i = 0; i < ENR_HIST; ++i) { enr_node_fp_hist[i] += other.enr_node_fp_hist[i]; }
    enr_node_fp_max_ns = enr_node_fp_max_ns < other.enr_node_fp_max_ns ? other.enr_node_fp_max_ns : enr_node_fp_max_ns;
    enr_backtracks += other.enr_backtracks;
    enr_rs_count += other.enr_rs_count; enr_rs_rounds += other.enr_rs_rounds; enr_rs_path_sum += other.enr_rs_path_sum;
    enr_rs_path_max = enr_rs_path_max < other.enr_rs_path_max ? other.enr_rs_path_max : enr_rs_path_max;
    enr_rs_restore_ns += other.enr_rs_restore_ns; enr_rs_fp_ns += other.enr_rs_fp_ns;
    enr_rs_cost_max_ns = enr_rs_cost_max_ns < other.enr_rs_cost_max_ns ? other.enr_rs_cost_max_ns : enr_rs_cost_max_ns;
    for(int i = 0; i < ENR_HIST; ++i) { enr_rs_hist[i] += other.enr_rs_hist[i]; }
    enr_sp_count += other.enr_sp_count; enr_sp_rounds += other.enr_sp_rounds; enr_sp_fp_ns += other.enr_sp_fp_ns;
    for(int i = 0; i < ENR_PH_N; ++i) { enr_ph_ns[i] += other.enr_ph_ns[i]; enr_ph_cnt[i] += other.enr_ph_cnt[i]; } //ENR STATS
  }

  template <class Alloc>
  CUDA void meet(const TimingStatistics<Alloc>& other) {
    timers.meet(other);
  }

#ifdef __CUDACC__
  __device__ cuda::std::chrono::system_clock::time_point start_timer_device() const {
    return timers.start_timer_device();
  }
#endif

  std::chrono::steady_clock::time_point start_timer_host() const {
    return timers.start_timer_host();
  }

#ifdef __CUDACC__
  __device__ cuda::std::chrono::system_clock::time_point stop_timer(Timer timer, cuda::std::chrono::system_clock::time_point start) {
    return timers.stop_timer(timer, start);
  }
#endif

  std::chrono::steady_clock::time_point stop_timer(Timer timer, std::chrono::steady_clock::time_point start) {
    return timers.stop_timer(timer, start);
  }

  /** Replace the value of the timer with `now - start`. */
  void update_timer(Timer timer, std::chrono::steady_clock::time_point start) {
    timers.update_timer(timer, start);
  }

  CUDA int64_t time_ms_of(Timer timer) const {
    return timers.time_ms_of(timer);
  }

  CUDA void print_stat(const char* name, const char* value) const {
    if(print_statistics) {
      printf("%%%%%%mzn-stat: %s=\"%s\"\n", name, value);
    }
  }

  CUDA void print_stat(const char* name, size_t value) const {
    if(print_statistics) {
      printf("%%%%%%mzn-stat: %s=%" PRIu64 "\n", name, value);
    }
  }

  CUDA void print_human_stat(int verbose, const char* name, size_t value) const {
    if(print_statistics) {
      printf("%%%%%%mzn-stat: %s=%" PRIu64 "\n", name, value);
      if(verbose) {
        printf("%%   [");
        size_t M = 1000 * 1000;
        if(value < M * 1000) {
          printf("%.2fM", static_cast<double>(value) / M);
        }
        else if(value < M * 1000 * 1000) {
          printf("%.2fG", static_cast<double>(value) / (M * 1000));
        }
        else {
          printf("%.2fT", static_cast<double>(value) / (M * 1000 * 1000));
        }
        printf("]\n");
      }
    }
  }

  CUDA void print_stat(const char* name, int value) const {
    if(print_statistics) {
      printf("%%%%%%mzn-stat: %s=%d\n", name, value);
    }
  }

  CUDA void print_stat_fp_iter(const char* name, size_t num_iterations, size_t value) const {
    if(print_statistics) {
      printf("%%%%%%mzn-stat: %s_fp_iter_%" PRIu64 "=%" PRIu64 "\n", name, num_iterations, value);
    }
  }

  CUDA void print_stat(const char* name, double value) const {
    if(print_statistics) {
      printf("%%%%%%mzn-stat: %s=%lf\n", name, value != value ? 0.0 : value); // check for NaN.
    }
  }

  CUDA void print_memory_statistics(int verbose, const char* key, size_t bytes) const {
    print_stat(key, bytes);
    if(verbose) {
      printf("%%   [");
      if(bytes < 1000 * 1000) {
        printf("%.2fKB", static_cast<double>(bytes) / 1000);
      }
      else if(bytes < 1000 * 1000 * 1000) {
        printf("%.2fMB", static_cast<double>(bytes) / (1000 * 1000));
      }
      else {
        printf("%.2fGB", static_cast<double>(bytes) / (1000 * 1000 * 1000));
      }
      printf("]\n");
    }
  }

private:
  CUDA double to_sec(int64_t dur) const {
    return (static_cast<double>(dur / 1000 / 1000) / 1000.);
  }

public:
  CUDA void print_block_timing_stat(const char* name, Timer timer) const {
    print_stat(name, to_sec(timers.time_of(timer) / num_blocks));
  }

  CUDA void print_timing_stat(const char* name, Timer timer) const {
    print_stat(name, to_sec(timers.time_of(timer)));
  }

  CUDA void print_mzn_statistics(int verbose = 0) const {
    print_stat("num_blocks", num_blocks);
    print_human_stat(verbose, "nodes", nodes);
    print_stat("failures", fails);
    print_stat("variables", variables);
    print_stat("propagators", constraints);
    print_stat("peakDepth", depth_max);
    print_timing_stat("initTime", Timer::PREPROCESSING);
    print_timing_stat("solveTime", Timer::OVERALL);
    print_stat("num_solutions", solutions);
    print_stat("eps_num_subproblems", eps_num_subproblems);
    print_stat("eps_solved_subproblems", eps_solved_subproblems);
    print_stat("eps_skipped_subproblems", eps_skipped_subproblems);
    print_stat("num_blocks_done", num_blocks_done);
    print_human_stat(verbose, "fixpoint_iterations", fixpoint_iterations);
    print_human_stat(verbose, "num_deductions", num_deductions);

    // Timing statistics
    print_stat("cumulative_time_block_sec", to_sec(cumulative_time_block));
    print_stat("deductions_per_block_second", num_deductions / num_blocks / to_sec(cumulative_time_block));
    print_block_timing_stat("solve_time", Timer::OVERALL);
    print_block_timing_stat("search_time", Timer::SEARCH);
    // print_block_timing_stat("split_time", Timer::SPLIT);
    // print_block_timing_stat("push_time", Timer::PUSH);
    // print_block_timing_stat("pop_time", Timer::POP);
    print_block_timing_stat("fixpoint_time", Timer::FIXPOINT);
    print_block_timing_stat("transfer_cpu2gpu_time", Timer::TRANSFER_CPU2GPU);
    print_block_timing_stat("transfer_gpu2cpu_time", Timer::TRANSFER_GPU2CPU);
    print_block_timing_stat("select_fp_functions_time", Timer::SELECT_FP_FUNCTIONS);
    print_block_timing_stat("wait_cpu_time", Timer::WAIT_CPU);
    print_block_timing_stat("dive_time", Timer::DIVE);
    print_timing_stat("best_obj_time", Timer::LATEST_BEST_OBJ_FOUND);
    print_timing_stat("first_block_idle_time", Timer::FIRST_BLOCK_IDLE);
    print_enr_restart_statistics(); //ENR STATS
    print_enr_statistics(); //ENR STATS
  }

  //ENR STATS -- whole function.
  CUDA void print_enr_restart_statistics() const {
    if(enr_tree_nodes > 0) {
      print_stat("enr_tree_nodes", enr_tree_nodes);
      print_stat("enr_tree_failures", enr_tree_fails);
      print_stat("enr_redundant_dive_nodes", nodes >= enr_tree_nodes ? nodes - enr_tree_nodes : size_t{0});
    }
    if(!enr_restart_mode) { return; }
    print_stat("enr_restarts", enr_restarts);
    print_stat("enr_restart_items_sum", enr_restart_items_sum);
    print_stat("enr_restart_items_max", enr_restart_items_max);
    print_stat("enr_restart_items_last", enr_restart_items_last);
    print_stat("enr_restart_subproblems_sum", enr_restart_subproblems_sum);
    print_stat("enr_restart_s_max", enr_restart_s_max);
    print_stat("enr_restart_truncated_blocks", enr_restart_truncated);
    print_stat("enr_restart_drain_ms_avg", enr_restarts == 0 ? 0.0 : static_cast<double>(enr_restart_drain_ns_sum) / static_cast<double>(enr_restarts) / 1e6);
    print_stat("enr_restart_drain_ms_max", static_cast<double>(enr_restart_drain_ns_max) / 1e6);
    print_stat("enr_restart_first_sec", enr_restart_first_ns < 0 ? -1.0 : static_cast<double>(enr_restart_first_ns) / 1e9);
    print_stat("enr_restart_last_sec", enr_restart_last_ns < 0 ? -1.0 : static_cast<double>(enr_restart_last_ns) / 1e9);
  }

  //ENR STATS -- whole function.
  CUDA void print_enr_statistics() const {
    if(!print_statistics) { return; }
    double life_ns = static_cast<double>(cumulative_time_block); // sum of block lifetimes
    double n = nodes == 0 ? 1.0 : static_cast<double>(nodes);
    double fp_ns = static_cast<double>(timers.time_of(Timer::FIXPOINT));
    //ENR STATS: three views. [kernel] = the solve kernel as a whole (num_blocks slots x kernel time),
    // [block] = one single block, [node] = one single node inside the block that computes it.
    double nb = num_blocks > 0 ? static_cast<double>(num_blocks) : 1.0;
    double kernel_ns = static_cast<double>(enr_life_block_max);  // longest block lifetime = duration of the kernel loop
    double cap_ns = kernel_ns * nb;                               // block-time available in the kernel
    double kernel_s = kernel_ns / 1e9;
    printf("%% ===== ENR statistics: [kernel] the solve kernel as a whole =====\n");
    print_stat("enr_kernel_time_sec", kernel_s);
    //ENR STATS: phases. Exclusive and contiguous, so with idle they sum to 100 % of num_blocks x kernel time.
    // Each pct is also the average over the blocks of (phase time of the block / kernel time).
    auto ph = [&](int p) { return static_cast<double>(enr_ph_ns[p]); };
    auto pc = [&](double ns) { return cap_ns > 0 ? 100.0 * ns / cap_ns : 0.0; };
    auto us = [&](double ns, size_t c) { return c == 0 ? 0.0 : ns / static_cast<double>(c) / 1000.0; };
    double idle_ns = static_cast<double>(enr_idle_ns_sum);
    double dive_all = ph(PH_SUBPB_SWITCH) + ph(PH_DIVE_FP) + ph(PH_DIVE_BRANCH);
    double search_all = ph(PH_SEARCH_FP) + ph(PH_SEARCH_BRANCH);
    double bt_all = ph(PH_BT_RESTORE) + ph(PH_BT_REDESCEND) + ph(PH_BT_FP);
    double ph_sum = dive_all + search_all + bt_all;
    printf("%% ===== ENR statistics: [phases] pct of (num_blocks x kernel time); dive + search + backtrack + idle = 100 =====\n");
    print_stat("enr_pct_dive", pc(dive_all));
    print_stat("enr_pct_dive_subproblem_switch", pc(ph(PH_SUBPB_SWITCH)));
    print_stat("enr_pct_dive_fixpoint", pc(ph(PH_DIVE_FP)));
    print_stat("enr_pct_dive_branch", pc(ph(PH_DIVE_BRANCH)));
    print_stat("enr_pct_search", pc(search_all));
    print_stat("enr_pct_search_fixpoint", pc(ph(PH_SEARCH_FP)));
    print_stat("enr_pct_search_branch", pc(ph(PH_SEARCH_BRANCH)));
    print_stat("enr_pct_backtrack", pc(bt_all));
    print_stat("enr_pct_backtrack_restore", pc(ph(PH_BT_RESTORE)));
    print_stat("enr_pct_backtrack_redescend", pc(ph(PH_BT_REDESCEND)));
    print_stat("enr_pct_backtrack_fixpoint", pc(ph(PH_BT_FP)));
    print_stat("enr_pct_idle", pc(idle_ns));
    print_stat("enr_pct_check_sum", pc(ph_sum + idle_ns));
    print_stat("enr_pct_fixpoint_all", pc(ph(PH_DIVE_FP) + ph(PH_SEARCH_FP) + ph(PH_BT_FP)));
    printf("%% [phases] counts (all blocks) and avg time of one occurrence (us):\n");
    print_stat("enr_dives", enr_ph_cnt[PH_SUBPB_SWITCH]);
    print_stat("enr_dive_us_avg", us(dive_all, enr_ph_cnt[PH_SUBPB_SWITCH]));
    //ENR STATS: every node = one propagate() = one fixpoint, of exactly one of these three kinds.
    // The times are of the whole propagate() call (fixpoint + leaf check), measured at the call site.
    size_t fp_cnt_all = enr_ph_cnt[PH_DIVE_FP] + enr_ph_cnt[PH_SEARCH_FP] + enr_ph_cnt[PH_BT_FP];
    double fp_ns_all = ph(PH_DIVE_FP) + ph(PH_SEARCH_FP) + ph(PH_BT_FP);
    printf("%% [fixpoints] dive + search + backtrack_last = total (= nodes); us = avg time of one propagate():\n");
    print_stat("enr_dive_fixpoints", enr_ph_cnt[PH_DIVE_FP]);
    print_stat("enr_dive_fixpoint_us_avg", us(ph(PH_DIVE_FP), enr_ph_cnt[PH_DIVE_FP]));
    print_stat("enr_search_fixpoints", enr_ph_cnt[PH_SEARCH_FP]);
    print_stat("enr_search_fixpoint_us_avg", us(ph(PH_SEARCH_FP), enr_ph_cnt[PH_SEARCH_FP]));
    print_stat("enr_backtrack_last_fixpoints", enr_ph_cnt[PH_BT_FP]);
    print_stat("enr_backtrack_last_fixpoint_us_avg", us(ph(PH_BT_FP), enr_ph_cnt[PH_BT_FP]));
    print_stat("enr_total_fixpoints", fp_cnt_all);
    print_stat("enr_total_fixpoint_us_avg", us(fp_ns_all, fp_cnt_all));
    printf("%% [backtrack] one backtrack = restore + redescend + the last fixpoint (us, avg):\n");
    print_stat("enr_backtracks", enr_ph_cnt[PH_BT_RESTORE]);
    print_stat("enr_backtrack_restore_us_avg", us(ph(PH_BT_RESTORE), enr_ph_cnt[PH_BT_RESTORE]));
    print_stat("enr_backtrack_redescend_us_avg", us(ph(PH_BT_REDESCEND), enr_ph_cnt[PH_BT_REDESCEND]));
    print_stat("enr_backtrack_fixpoint_us_avg", us(ph(PH_BT_FP), enr_ph_cnt[PH_BT_FP]));
    print_stat("enr_backtrack_whole_us_avg", us(bt_all, enr_ph_cnt[PH_BT_RESTORE]));
    printf("%% ===== ENR statistics: [kernel] rates and blocks done =====\n");
    print_stat("enr_kernel_nodes_per_sec", kernel_s > 0 ? static_cast<double>(nodes) / kernel_s : 0.0);
    print_stat("enr_whole_gpu_deductions_per_sec", kernel_s > 0 ? static_cast<double>(num_deductions) / kernel_s : 0.0);
    // Blocks that ran out of work (not stopped by the timeout): how many, and when the first and the last one exited.
    print_stat("enr_blocks_done", enr_done_blocks);
    print_stat("enr_kernel_pct_blocks_done", 100.0 * static_cast<double>(enr_done_blocks) / nb);
    print_stat("enr_kernel_first_block_done_sec", enr_first_done_ns < 0 ? -1.0 : static_cast<double>(enr_first_done_ns) / 1e9);
    print_stat("enr_kernel_first_block_done_pct_of_kernel", (enr_first_done_ns < 0 || kernel_ns <= 0) ? -1.0 : 100.0 * static_cast<double>(enr_first_done_ns) / kernel_ns);
    print_stat("enr_kernel_last_block_done_sec", enr_last_done_ns < 0 ? -1.0 : static_cast<double>(enr_last_done_ns) / 1e9);
    print_stat("enr_kernel_last_block_done_pct_of_kernel", (enr_last_done_ns < 0 || kernel_ns <= 0) ? -1.0 : 100.0 * static_cast<double>(enr_last_done_ns) / kernel_ns);
    printf("%% enr_kernel idle timeline (kernel time cut in %d slices; idle = %% of the %d block slots empty during the slice):\n", ENR_TL, num_blocks);
    for(int k = 0; k < ENR_TL; ++k) {
      printf("%%   %3d-%3d%% of kernel = [%10.3f, %10.3f) s  idle %6.2f%%   busy blocks at end %4d\n",
        100 * k / ENR_TL, 100 * (k + 1) / ENR_TL, kernel_s * k / ENR_TL, kernel_s * (k + 1) / ENR_TL, enr_idle_pct_slice[k], enr_busy_end_slice[k]);
    }
    printf("%% ===== ENR statistics: [block] one single block (avg over the blocks, min/max over the blocks) =====\n");
    print_stat("enr_block_lifetime_sec_avg", life_ns / nb / 1e9);
    print_stat("enr_block_lifetime_sec_min", static_cast<double>(enr_life_block_min) / 1e9);
    print_stat("enr_block_lifetime_sec_max", static_cast<double>(enr_life_block_max) / 1e9);
    print_stat("enr_block_nodes_avg", static_cast<double>(nodes) / nb);
    print_stat("enr_block_nodes_min", enr_nodes_block_min);
    print_stat("enr_block_nodes_max", enr_nodes_block_max);
    print_stat("enr_block_subproblems_avg", static_cast<double>(eps_solved_subproblems) / nb);
    print_stat("enr_block_subproblems_min", enr_subpb_block_min);
    print_stat("enr_block_subproblems_max", enr_subpb_block_max);
    print_stat("enr_block_nodes_per_sec_avg", life_ns > 0 ? n / (life_ns / 1e9) : 0.0);
    print_stat("enr_block_deductions_per_sec_avg", life_ns > 0 ? static_cast<double>(num_deductions) / (life_ns / 1e9) : 0.0);
    printf("%% ===== ENR statistics: [node] one single node, in the block that computes it (avg over all nodes) =====\n");
    print_stat("enr_node_time_us_avg", life_ns / n / 1000.0);
    print_stat("enr_node_fixpoint_us_avg", fp_ns / n / 1000.0);
    print_stat("enr_node_fixpoint_us_max", static_cast<double>(enr_node_fp_max_ns) / 1000.0);
    print_stat("enr_node_rounds_avg", static_cast<double>(fixpoint_iterations) / n);
    print_stat("enr_node_deductions_avg", static_cast<double>(num_deductions) / n);
    printf("%% enr_node fixpoint-time histogram (us), percentage of the nodes:\n");
    for(int i = 0; i < ENR_HIST; ++i) {
      if(enr_node_fp_hist[i] == 0) continue;
      long lo = i == 0 ? 0 : (1L << (i - 1));
      if(i == ENR_HIST - 1) { printf("%%   [%6ld, +inf ) us: %" PRIu64 " (%.2f%%)\n", lo, enr_node_fp_hist[i], 100.0 * enr_node_fp_hist[i] / n); }
      else { printf("%%   [%6ld, %6ld) us: %" PRIu64 " (%.2f%%)\n", lo, 1L << i, enr_node_fp_hist[i], 100.0 * enr_node_fp_hist[i] / n); }
    }
    //ENR STATS: backtrack cost histogram (enr_record_backtrack: restore_ns = restore + redescend, fp_ns = pure fixpoint).
    double rc = enr_rs_count == 0 ? 1.0 : static_cast<double>(enr_rs_count);
    printf("%% ===== ENR statistics: [backtrack] cost of one backtrack (us, restore + redescend + pure fixpoint), percentage of the backtracks =====\n");
    for(int i = 0; i < ENR_HIST; ++i) {
      if(enr_rs_hist[i] == 0) continue;
      long lo = i == 0 ? 0 : (1L << (i - 1));
      if(i == ENR_HIST - 1) { printf("%%   [%6ld, +inf ) us: %" PRIu64 " (%.2f%%)\n", lo, enr_rs_hist[i], 100.0 * enr_rs_hist[i] / rc); }
      else { printf("%%   [%6ld, %6ld) us: %" PRIu64 " (%.2f%%)\n", lo, 1L << i, enr_rs_hist[i], 100.0 * enr_rs_hist[i] / rc); }
    }
  }

  CUDA void print_mzn_end_stats() const {
    if(!print_statistics) { return; }
    printf("%%%%%%mzn-stat-end\n");
  }

  CUDA void print_mzn_objective(const auto& obj, bool is_minimization) const {
    if(!print_statistics) { return; }
    printf("%%%%%%mzn-stat: objective=");
    if(is_minimization) {
      obj.lb().print();
    }
    else {
      obj.ub().print();
    }
    printf("\n");
  }

  CUDA void print_mzn_separator() const {
    printf("----------\n");
  }

  CUDA void print_mzn_final_separator() const {
    if(solutions > 0) {
      if(exhaustive) {
        printf("==========\n");
      }
    }
    else {
      assert(solutions == 0);
      if(exhaustive) {
        printf("=====UNSATISFIABLE=====\n");
      }
      else if(optimization) {
        printf("=====UNBOUNDED=====\n");
      }
      else {
        printf("=====UNKNOWN=====\n");
      }
    }
  }
};

#endif
