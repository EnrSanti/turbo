// Copyright 2025 Pierre Talbot

#ifndef TURBO_BAREBONES_DIVE_AND_SOLVE_HPP
#define TURBO_BAREBONES_DIVE_AND_SOLVE_HPP

#include "common_solving.hpp"
#include "memory_gpu.hpp"
#include "lala/light_branch.hpp"
#include <mutex>
#include <thread>
#include <chrono>

/** This is required in order to guess the usage of global memory, and increase the CUDA default limit. */
#define MAX_SEARCH_DEPTH 10000

namespace bt = ::battery;

/**
 * The full GPU version (`gpu_dive_and_solve`) is not compiling on modern GPU hardware (SM >= 9) due to the kernel being too large.
 * We circuvanted this issue by creating an hybrid version where only propagation is executed on the GPU (`hybrid_dive_and_solve`).
 * This has the disadvantage of memory transfers between CPU and GPU and synchronization overheads.
 * We propose a new "barebones" version which contains less abstractions than the GPU and hybrid versions, but have the same functionalities.
 * In particular, we directly implement the branch-and-bound algorithm here and avoid using `lala::BAB` and `lala::SearchTree` which are nice from a software engineering perspective but bring significant overhead.
 * This version is intended to reach the best possible performance.
 *
 * Terminology:
 *  * unified data: data available to both the CPU and GPU.
 *  * block data: data used within a single block.
 *  * grid data: data shared among all blocks in the grid.
 */

#ifdef __CUDACC__

#include <cuda/std/chrono>
#include <cuda/semaphore>
#include <cuda/atomic>               //ENR CODE
#include <cooperative_groups.h>      //ENR CODE

#endif

namespace barebones {

#ifdef __CUDACC__
#ifndef TURBO_IPC_ABSTRACT_DOMAIN

/** `ConcurrentAllocator` allocates memory available both on CPU and GPU. For non-Linux systems such as Windows pinned memory must be used (see PR #19). */
#ifdef NO_CONCURRENT_MANAGED_MEMORY
  using ConcurrentAllocator = bt::pinned_allocator;
#else
  using ConcurrentAllocator = bt::managed_allocator;
#endif

using GridCP = AbstractDomains<Itv,
  bt::statistics_allocator<ConcurrentAllocator>,
  bt::statistics_allocator<UniqueLightAlloc<ConcurrentAllocator, 0>>,
  bt::statistics_allocator<UniqueLightAlloc<ConcurrentAllocator, 1>>>;

/** Data shared between CPU and GPU. */
struct UnifiedData {
  /** The root node of the problem, useful to backtrack when solving a new subproblem.
   * Also contains the shared information such as statistics and solver configuration.
   */
  GridCP root;

  /** Stop signal from the CPU because of a timeout or CTRL-C. */
  cuda::std::atomic_flag stop;

  /** The memory configuration of each block. */
  MemoryConfig mem_config;

  UnifiedData(const CP<Itv>& cp, MemoryConfig mem_config)
   : root(GridCP::tag_gpu_block_copy{}, false, cp)
   , stop(false)
   , mem_config(mem_config)
  {
    size_t num_subproblems = 1;
    num_subproblems <<= root.config.subproblems_power;
    root.stats.eps_num_subproblems = num_subproblems;
  }
};

struct GridData;
using IStore = VStore<Itv, bt::pool_allocator>;
using IProp = PIR<IStore, bt::pool_allocator>;
using UB = ZUB<typename Itv::LB::value_type, bt::atomic_memory_grid>;
using strategies_type = bt::vector<StrategyType<bt::global_allocator>, bt::global_allocator>;

//ENR STATS -- begin: used only for the restart times (raise, drain, first/last).
/** Device clock in nanoseconds (globaltimer: the same clock on every SM). */
__device__ INLINE int64_t enr_restart_now_ns() {
  return cuda::std::chrono::duration_cast<cuda::std::chrono::nanoseconds>(
    cuda::std::chrono::system_clock::now().time_since_epoch()).count();
}
//ENR STATS -- end

//ENR CODE -- begin
/** An open node left by a block stopped by a restart: the path of block `snap` (its snapshot) down to level `k`,
 * then child `c` of the decision at level `k` (0 = children[0], 1 = children[1]). (8 bytes) */
struct EnrFrontierItem {
  int snap;
  unsigned int k : 31;
  unsigned int c : 1;
};
//ENR CODE -- end

/** Data private to a single block. */
struct BlockData {
  /** The store of variables at the root of the current subproblem. */
  abstract_ptr<VStore<Itv, bt::global_allocator>> root_store;

  /** The best solution found so far in this block. */
  abstract_ptr<VStore<Itv, bt::global_allocator>> best_store;

  /** The current store of variables.
   * We use a `pool_allocator`, this allows to easily switch between global memory and shared memory, if the store of variables can fit inside.
   * */
  abstract_ptr<IStore> store;

  /** The propagators implemented as an array of bytecodes.
   * Similarly, the propagators can be stored in the global or shared memory.
   * If the propagators do not fit in shared memory, the array of propagators is shared among all blocks.
   * It is possible because the propagators are state-less, we avoid duplicating them in each block.
   * */
  abstract_ptr<IProp> iprop;

  /** The statistics of the current block. */
  Statistics<bt::global_allocator> stats;

  /** The path from `UnifiedData::root` to the current subproblem `root_store`. */
  size_t subproblem_idx;

  /** The best bound found so far by this block.
   * We always seek to minimize.
   * Invariant: `best_bound == best_store.project(obj_var).lb()`
   */
  UB best_bound;

  /** The current strategy being used to split the store.
   * It is an index into `GridData::strategies`.
   */
  int current_strategy;

  /** The next unassigned variable in the current strategy.
   * It is an index into `GridData::strategies.vars`.
   */
  int next_unassigned_var;

  /** On backtracking, the value to restore `current_strategy`. */
  int snapshot_root_strategy;

  /** On backtracking, the value to restore `next_unassigned_var`. */
  int snapshot_next_unassigned_var;

  /** The decision taken when exploring the tree. */
  bt::vector<LightBranch<Itv>, bt::global_allocator> decisions;

  /** Current depth of the search tree. */
  int depth;

  /** A timer used for computing time statistics. */
  cuda::std::chrono::system_clock::time_point timer;

  /** A timer used for computing diving VS search time statistics. */
  cuda::std::chrono::system_clock::time_point dive_timer;

  /** The time at which the kernel was started, useful to compute the time of the best bound. */
  cuda::std::chrono::system_clock::time_point start_time;

  //ENR STATS -- begin
  bool enr_after_restore = false;   // the next fixpoint is the one right after a backtrack (restart).
  int64_t enr_restore_ns = 0;       // restore + redescend time of the last backtrack.
  int enr_restore_path = 0;         // number of decisions re-applied by the last backtrack.
  cuda::std::chrono::system_clock::time_point enr_restore_start;
  // Phase chain: each ENR_PH(P, now, c) charges (now - enr_ph_t) to phase P, then enr_ph_t = now.
  // The charges are contiguous from start_time to the end of the block, so the phases sum to the block lifetime.
  cuda::std::chrono::system_clock::time_point enr_ph_t;
  // Waits at the restart barriers (from the arrival at S1 to the exit of S3), ns relative to start_time, for the idle timeline.
  static constexpr int ENR_MAX_WAITS = 256;
  int64_t enr_wait_from[ENR_MAX_WAITS];
  int64_t enr_wait_to[ENR_MAX_WAITS];
  int enr_n_waits = 0;
  int64_t enr_wait_ns = 0;          // total wait (also counts the waits beyond ENR_MAX_WAITS).
  int64_t enr_wait_begin = 0;
  //ENR STATS -- end

  
  //ENR CODE -- begin
  /** Snapshot of `root_store` and `decisions[0..depth)` taken when this block stops because of a restart.
   * It is read-only during the next round: any block rebuilds this block's open nodes from it (step C'). */
  //populated  when harvesting by each active block
  abstract_ptr<VStore<Itv, bt::global_allocator>> enr_snap_store; //subproblem root 
  bt::vector<LightBranch<Itv>, bt::global_allocator> enr_snap_dec; //subproblem stack when stopped
  int enr_snap_root_strategy = 0;
  //ENR CODE -- end


  //ENR STATS -- begin
  /** True if the next `propagate()` visits a node for the first time in the search tree (statistics `enr_tree_nodes`).
   * False for a dive node already visited by a subproblem with a smaller index (same prefix). */
  bool enr_in_tree = true;
  //ENR STATS -- end

  __device__ BlockData()
   : subproblem_idx(0)
   , current_strategy(0)
   , next_unassigned_var(0)
   , decisions(5000)
   , depth(0)
  {}

  __device__ void allocate(const UnifiedData& unified_data, const GridData& grid_data, unsigned char* shared_mem) {
    if(threadIdx.x == 0) {
      subproblem_idx = blockIdx.x;
      const MemoryConfig& mem_config = unified_data.mem_config;
      const auto& u_store = *(unified_data.root.store);
      const auto& u_iprop = *(unified_data.root.iprop);
      bt::pool_allocator shared_mem_pool(mem_config.make_shared_pool(shared_mem));
      bt::pool_allocator store_allocator(mem_config.make_store_pool(shared_mem_pool));
      bt::pool_allocator prop_allocator(mem_config.make_prop_pool(shared_mem_pool));
      root_store = bt::make_shared<VStore<Itv, bt::global_allocator>, bt::global_allocator>(u_store);
      best_store = bt::make_shared<VStore<Itv, bt::global_allocator>, bt::global_allocator>(u_store);
      store = bt::allocate_shared<IStore, bt::pool_allocator>(store_allocator, u_store, store_allocator);
      iprop = bt::allocate_shared<IProp, bt::pool_allocator>(prop_allocator, u_iprop, store, prop_allocator);
      //ENR CODE -- begin: the snapshot used by the restarts (only with -restart / -restart_each_block).
      // The flag is read from the config: GridData is still incomplete here, and grid_data.enr_restart_enabled is this same value.
      if(unified_data.root.config.enr_restart_on()) {
        enr_snap_store = bt::make_shared<VStore<Itv, bt::global_allocator>, bt::global_allocator>(u_store);
        enr_snap_dec.resize(decisions.size());
      }
      //ENR CODE -- end
    }
  }

  /** We must deallocate store and iprop inside the kernel because they might be initialized in shared memory. */
  __device__ void deallocate_shared_data() {
    if(threadIdx.x == 0) {
      // NOTE: .reset() does not work because it does not reset the allocator, which is itself allocated in global memory.
      store = abstract_ptr<IStore>();
      iprop = abstract_ptr<IProp>();
    }
  }

  /** Add a new decision on the `decisions` stack and increase depth.
   * \param has_changed: A Boolean in shared memory.
   * \param strategies: A sequence of strategies.
   * \precondition: We must not be on a leaf node.
   */
  __device__ INLINE void split(bool& has_changed, const strategies_type& strategies) {
    using LB2 = typename Itv::LB::local_type;
    using UB2 = typename Itv::UB::local_type;
    __shared__ local::ZUB idx;
    decisions[depth].var = AVar{};
    int currentDepth = depth;
    for(int i = current_strategy; i < strategies.size(); ++i) {
      switch(strategies[i].var_order) {
        case VariableOrder::RANDOM:
        case VariableOrder::INPUT_ORDER: {
          input_order_split(has_changed, idx, strategies[i]);
          break;
        }
        case VariableOrder::FIRST_FAIL: {
          lattice_smallest_split(has_changed, idx, strategies[i],
            [](const Itv& u) { return UB2(u.ub().value() - u.lb().value()); });
          break;
        }
        case VariableOrder::ANTI_FIRST_FAIL: {
          lattice_smallest_split(has_changed, idx, strategies[i],
            [](const Itv& u) { return LB2(u.ub().value() - u.lb().value()); });
          break;
        }
        case VariableOrder::LARGEST: {
          lattice_smallest_split(has_changed, idx, strategies[i],
            [](const Itv& u) { return LB2(u.ub().value()); });
          break;
        }
        case VariableOrder::SMALLEST: {
          lattice_smallest_split(has_changed, idx, strategies[i],
            [](const Itv& u) { return UB2(u.lb().value()); });
          break;
        }
        default: assert(false);
      }
      __syncthreads();
      // If we could find a variable with the current strategy, we return.
      if(!decisions[currentDepth].var.is_untyped()) {
        return;
      }
      if(threadIdx.x == 0) {
        current_strategy++;
        next_unassigned_var = 0;
      }
    }
    // `input_order_split` and `lattice_smallest_split` have a `__syncthreads()` before reading next_unassigned_var.
  }

  /** Select the next unassigned variable with a finite interval in the array `strategy.vars()` or `store` if the previous one is empty.
   * We ignore infinite variables as splitting on them do not guarantee termination.
   * \param has_changed is a Boolean in shared memory.
   * \param idx is a decreasing integer in shared memory.
   */
  __device__ INLINE void input_order_split(bool& has_changed, local::ZUB& idx, const StrategyType<bt::global_allocator>& strategy)
  {
    bool split_in_store = strategy.vars.empty();
    int n = split_in_store ? store->vars() : strategy.vars.size();
    if(threadIdx.x == 0) {
      has_changed = true;
      idx = n;
    }
    __syncthreads();
    while(has_changed) {
      __syncthreads();
      // int n = idx.value();
      if(threadIdx.x == 0) {
        has_changed = false;
      }
      __syncthreads();
      for(int i = next_unassigned_var + threadIdx.x; i < n; i += blockDim.x) {
        const auto& dom = (*store)[split_in_store ? i : strategy.vars[i].vid()];
        if(dom.lb().value() != dom.ub().value() && !dom.lb().is_top() && !dom.ub().is_top()) {
          if(idx.meet(local::ZUB(i))) {
            has_changed = true;
          }
        }
      }
      __syncthreads();
    }
    if(threadIdx.x == 0) {
      next_unassigned_var = idx.value();
      if(next_unassigned_var != n) {
        push_decision(strategy.val_order, split_in_store ? AVar{store->aty(), next_unassigned_var} : strategy.vars[next_unassigned_var]);
      }
    }
  }

  /** Given an array of variable, select the variable `x` with the smallest value `f(store[x])` where "smallest" is defined according to the lattice order of the return type of `f`.
   * \param has_changed is a Boolean in shared memory.
   * \param idx is a decreasing integer in shared memory.
   * */
  template <class F>
  __device__ INLINE void lattice_smallest_split(bool& has_changed, local::ZUB& idx,
    const StrategyType<bt::global_allocator>& strategy, F f)
  {
    using T = decltype(f(Itv{}));
    __shared__ T value;
    bool split_in_store = strategy.vars.empty();
    int n = split_in_store ? store->vars() : strategy.vars.size();
    __syncthreads();
    if(threadIdx.x == 0) {
      has_changed = true;
      value = T::top();
      idx = n;
    }
    __syncthreads();
    /** This fixpoint loop seeks for the smallest `x` according to `f(x)` and the next unassigned variable. */
    while(has_changed) {
      __syncthreads();
      if(threadIdx.x == 0) {
        has_changed = false;
      }
      __syncthreads();
      for(int i = next_unassigned_var + threadIdx.x; i < n; i += blockDim.x) {
        const auto& dom = (*store)[split_in_store ? i : strategy.vars[i].vid()];
        if(dom.lb().value() != dom.ub().value() && !dom.lb().is_top() && !dom.ub().is_top()) {
          if(value.meet(f(dom))) {
            has_changed = true;
          }
          if(idx.meet(local::ZUB(i))) {
            has_changed = true;
          }
        }
      }
      __syncthreads();
    }
    /** If we found a value, we traverse again the variables' array to find its index. */
    if(!value.is_top()) {
      __syncthreads();
      if(threadIdx.x == 0) {
        next_unassigned_var = idx.value();
        idx = n;
        has_changed = true;
      }
      __syncthreads();
      // This fixpoint loop is not strictly necessary.
      // We keep it for determinism: the variable with the smallest index is selected first.
      while(has_changed) {
        // int n = idx.value();
        __syncthreads();
        has_changed = false;
        __syncthreads();
        for(int i = next_unassigned_var + threadIdx.x; i < n; i += blockDim.x) {
          const auto& dom = (*store)[split_in_store ? i : strategy.vars[i].vid()];
          if(dom.lb().value() != dom.ub().value() && !dom.lb().is_top() && !dom.ub().is_top() && f(dom) == value) {
            if(idx.meet(local::ZUB(i))) {
              has_changed = true;
            }
          }
        }
        __syncthreads();
      }
      assert(idx.value() < n);
      if(threadIdx.x == 0) {
        if(split_in_store) {
          push_decision(strategy.val_order, AVar{store->aty(), idx.value()});
        }
        else {
          push_decision(strategy.val_order, strategy.vars[idx.value()]);
        }
      }
    }
  }

  /** Push a new decision onto the decisions stack.
   *  \precondition The domain of the variable `var` must not be empty, be a singleton or contain infinite bounds.
   *  \precondition Must be executed by thread 0 only.
  */
  __device__ INLINE void push_decision(ValueOrder val_order, AVar var) {
    using value_type = typename Itv::LB::value_type;
    assert(threadIdx.x == 0);
    decisions[depth].var = var;
    decisions[depth].current_idx = -1;
    const auto& dom = store->project(decisions[depth].var);
    assert(dom.lb().value() != dom.ub().value());
    switch(val_order) {
      case ValueOrder::MIN: {
        decisions[depth].children[0] = Itv(dom.lb().value());
        decisions[depth].children[1] = Itv(dom.lb().value() + value_type{1}, dom.ub());
        break;
      }
      case ValueOrder::MAX: {
        decisions[depth].children[0] = Itv(dom.ub().value());
        decisions[depth].children[1] = Itv(dom.lb(), dom.ub().value() - value_type{1});
        break;
      }
      case ValueOrder::SPLIT: {
        auto mid = dom.lb().value() +  (dom.ub().value() - dom.lb().value()) / value_type{2};
        decisions[depth].children[0] = Itv(dom.lb(), mid);
        decisions[depth].children[1] = Itv(mid + value_type{1}, dom.ub());
        break;
      }
      case ValueOrder::REVERSE_SPLIT: {
        auto mid = dom.lb().value() +  (dom.ub().value() - dom.lb().value()) / value_type{2};
        decisions[depth].children[0] = Itv(mid + value_type{1}, dom.ub());
        decisions[depth].children[1] = Itv(dom.lb(), mid);
        break;
      }
      // ValueOrder::MEDIAN is not possible with interval.
      default: assert(false);
    }
    /** Ropes are a mechanism for fast backtracking.
     * The rope of a left node is always the depth of the right node (also its depth), because after completing the exploration of the left subtree, we must visit the right subtree (rooted at the current depth).
     * The rope of the right node is inherited from its parent, we set -1 if there is no next node to visit.
     */
    decisions[depth].ropes[0] = depth + 1;
    decisions[depth].ropes[1] = depth > 0 ? decisions[depth-1].ropes[decisions[depth-1].current_idx] : -1;
    ++depth;
    // printf("depth(%d), var = %d, children = [%d, %d] | [%d, %d], ropes = [%d, %d]\n",
    //   depth, decisions[depth-1].var.vid(),
    //   (int)decisions[depth-1].children[0].lb().value(), (int)decisions[depth-1].children[0].ub().value(),
    //   (int)decisions[depth-1].children[1].lb().value(), (int)decisions[depth-1].children[1].ub().value(),
    //   decisions[depth-1].ropes[0], decisions[depth-1].ropes[1]);
    // Reallocate decisions if needed.
    if(decisions.size() == depth) {
      printf("resize to %d\n", (int)decisions.size() * 2);
      decisions.resize(decisions.size() * 2);
    }
  }
};

/** Data shared among all blocks. */
struct GridData {
  /** The private data of each block. */
  bt::vector<BlockData, bt::global_allocator> blocks;

  /** We generate the subproblems lazily.
   * Suppose we generate `2^3` subproblems, we represent the first subproblem as `000`, the second as `001`, the third as `010`, and so on.
   * A `0` means to turn left in the search tree, and a `1` means to turn right.
   * Incrementing this integer will generate the path of the next subproblem.
   */
  ZLB<size_t, bt::atomic_memory_grid> next_subproblem;

  /** This is an approximation of the best bound found so far, globally, across all threads.
   * It is not necessarily the true best bound at each time `t`.
   * The true best bound is obtained by `meet` over all block's best bounds.
   * It is used to share information among blocks.
   * We always seek to minimize.
   */
  UB appx_best_bound;

  /** Due to multithreading, we must protect `stdout` when printing.
   * The model of computation in this work is lock-free, but it seems unavoidable for printing.
  */
  cuda::binary_semaphore<cuda::thread_scope_device> print_lock;

  /** A specific strategy is used for the subproblem decomposition during the diving phase. */
  bool has_eps_strategy;

  /** The search strategy is immutable and shared among the blocks. */
  strategies_type search_strategies;

  /** The objective variable to minimize.
   * Maximization problem are transformed into minimization problems by negating the objective variable.
   * Equal to -1 if the problem is a satisfaction problem.
   */
  AVar obj_var;

  //ENR CODE -- begin
  bool enr_restart_enabled;        // -restart P > 0 or -restart_each_block: the kernel runs rounds separated by grid-wide barriers (cooperative launch).
  int enr_restart_threshold;       // the idle block that brings the idle count to this value raises the flag: 1 with -restart_each_block, else ceil(P * B / 100).
  size_t enr_restart_target;       // T = subproblems_factor * B (-subfactor): a round has at least T subproblems (n * 2^s >= T).
  int enr_items_per_block;         // quota of open nodes one block may publish at one restart.
  bt::vector<EnrFrontierItem, bt::global_allocator> enr_items;     // open nodes of the current round, compact in [0, n).
  cuda::atomic<int, cuda::thread_scope_device> enr_restart_flag; // raised by the idle block that reaches the threshold, cleared by block 0 between S1 and S3.
  cuda::atomic<int, cuda::thread_scope_device> enr_idle_blocks;  // blocks that ran out of work in the current round.
  cuda::atomic<int, cuda::thread_scope_device> enr_n_items[2];   // item counter of restart r is enr_n_items[r & 1].
  bool enr_host_stop;              // the host stop flag, read once by block 0 between S1 and S3.
  //ENR CODE -- end
  //ENR STATS -- begin: copied into the root statistics by `reduce_blocks`.
  int64_t enr_raise_ns;            // device clock when the flag was raised.
  int64_t enr_kernel_start_ns;
  int64_t enr_pending_drain_ns;    // flag -> S1 of the last barrier (-1 if the barrier was not a restart).
  int64_t enr_pending_raise_ns;
  size_t enr_st_restarts;          // restarts that redistributed work (n > 0).
  size_t enr_st_items_sum, enr_st_items_max, enr_st_items_last, enr_st_subproblems_sum;
  int enr_st_s_max;
  cuda::atomic<int, cuda::thread_scope_device> enr_st_truncated;
  int64_t enr_st_drain_sum, enr_st_drain_max, enr_st_first_ns, enr_st_last_ns;
  //ENR STATS -- end

  __device__ GridData(const GridCP& root)
   : blocks(root.stats.num_blocks)
   , next_subproblem(root.stats.num_blocks)
   , print_lock(1)
   , has_eps_strategy(root.config.eps_var_order != "default")
   , search_strategies(root.split->strategies_())
   , obj_var(root.minimize_obj_var)
   , enr_restart_enabled(root.config.enr_restart_on()) //ENR CODE
   , enr_items_per_block(1024) //ENR CODE
   , enr_items(root.config.enr_restart_on() ? static_cast<size_t>(root.stats.num_blocks) * 1024 : 0) //ENR CODE
  {
    //ENR CODE -- begin
    int B = root.stats.num_blocks;
    enr_restart_threshold = root.config.enr_restart_each_block ? 1 : static_cast<int>((root.config.enr_restart_idle_pct * B + 99) / 100);
    if(enr_restart_threshold < 1) { enr_restart_threshold = 1; }
    enr_restart_target = root.config.subproblems_factor * static_cast<size_t>(B);  //how many subproblems to generate in a round at least
    enr_restart_flag.store(0); enr_idle_blocks.store(0); enr_n_items[0].store(0); enr_n_items[1].store(0);
    enr_host_stop = false;
    //ENR CODE -- end
    //ENR STATS -- begin
    enr_raise_ns = 0; enr_kernel_start_ns = enr_restart_now_ns();
    enr_pending_drain_ns = -1; enr_pending_raise_ns = 0;
    enr_st_restarts = 0; enr_st_items_sum = 0; enr_st_items_max = 0; enr_st_items_last = 0; enr_st_subproblems_sum = 0;
    enr_st_s_max = 0; enr_st_truncated.store(0);
    enr_st_drain_sum = 0; enr_st_drain_max = 0; enr_st_first_ns = -1; enr_st_last_ns = -1;
    //ENR STATS -- end
  }
};




MemoryConfig configure_gpu_barebones(CP<Itv>&);
__global__ void initialize_global_data(UnifiedData*, bt::unique_ptr<GridData, bt::global_allocator>*);
__global__ void gpu_barebones_solve(UnifiedData*, GridData*);
template <class FPEngine>
__device__ INLINE void propagate(UnifiedData& unified_data, GridData& grid_data, BlockData& block_data,
   FPEngine& fp_engine, bool& stop, bool& enr_restart, bool& has_changed, bool& is_leaf_node); //ENR CODE: + bool& enr_restart
__global__ void reduce_blocks(UnifiedData*, GridData*);
__global__ void deallocate_global_data(bt::unique_ptr<GridData, bt::global_allocator>*);

//main entry for gpu

void barebones_dive_and_solve(const Configuration<battery::standard_allocator>& config) {

  if(config.print_intermediate_solutions) {
    printf("%% WARNING: -arch barebones is incompatible with -i and -a (it cannot print intermediate solutions).\n");
  }
 
  auto start = std::chrono::steady_clock::now();
  
  //check if the GPU supports unified memory 
  check_support_managed_memory();
  check_support_concurrent_managed_memory();
 
  // We start with some preprocessing to reduce the number of variables and constraints
  CP<Itv> cp(config);
  
  // load file and preprocess to TCN and simplify 
  cp.preprocess();
  if(cp.iprop->is_bot()) {
    cp.print_final_solution();
    cp.print_mzn_statistics();
    return;
  }




  // set up gpu
  MemoryConfig mem_config = configure_gpu_barebones(cp);
  // Allocate unified data and grid data on the GPU.
  auto unified_data = bt::make_unique<UnifiedData, ConcurrentAllocator>(cp, mem_config);
  auto grid_data = bt::make_unique<bt::unique_ptr<GridData, bt::global_allocator>, ConcurrentAllocator>();
  initialize_global_data<<<1,1>>>(unified_data.get(), grid_data.get());
  CUDAEX(cudaDeviceSynchronize());




  
  /** We wait that either the solving is interrupted, or that all threads have finished. */
  /** Block the signal CTRL-C to notify the threads if we must exit. */
  block_signal_ctrlc();

  if(cp.config.enr_restart_on()) { //ENR CODE: cooperative launch, required by grid.sync().
    //get data as before
    UnifiedData* ud_ptr = unified_data.get();
    GridData* gd_ptr = grid_data->get(); 
    void* kernel_args[] = { &ud_ptr, &gd_ptr };
    CUDAEX(cudaLaunchCooperativeKernel((void*) gpu_barebones_solve, 
      dim3(static_cast<unsigned int>(cp.stats.num_blocks)), dim3(CUDA_THREADS_PER_BLOCK),
      kernel_args, mem_config.shared_bytes, 0)); //cooperative launch
  } 
  else { //ENR CODE (normal launch)
  gpu_barebones_solve
    <<<static_cast<unsigned int>(cp.stats.num_blocks),
      CUDA_THREADS_PER_BLOCK,
      mem_config.shared_bytes>>>
    (unified_data.get(), grid_data->get());
  } //ENR CODE


  auto now = std::chrono::steady_clock::now();
  int64_t time_to_kernel_start = std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count();
  bool interrupted = wait_solving_ends(unified_data->stop, unified_data->root, start);
  
  //for stats only
  CUDAEX(cudaDeviceSynchronize());
  reduce_blocks<<<1,1>>>(unified_data.get(), grid_data->get());
  CUDAEX(cudaDeviceSynchronize());
  auto& uroot = unified_data->root;

  if(uroot.stats.solutions > 0) {
    // We add the time before the kernel starts to the time needed to find the best bound.
    uroot.stats.timers.time_of(Timer::LATEST_BEST_OBJ_FOUND) += time_to_kernel_start;
    if(uroot.stats.timers.time_of(Timer::FIRST_BLOCK_IDLE) != 0) {
      uroot.stats.timers.time_of(Timer::FIRST_BLOCK_IDLE) += time_to_kernel_start;
    }
    cp.print_solution(*uroot.best);
  }
  

  uroot.stats.print_mzn_final_separator();
  if(uroot.config.print_statistics) {
    uroot.config.print_mzn_statistics();
    uroot.stats.print_mzn_statistics(uroot.config.verbose_solving);
    if(uroot.bab->is_optimization() && uroot.stats.solutions > 0) {
      uroot.stats.print_mzn_objective(uroot.best->project(uroot.bab->objective_var()), uroot.bab->is_minimization());
    }
    uroot.stats.print_mzn_end_stats();
  }


  deallocate_global_data<<<1,1>>>(grid_data.get());
  CUDAEX(cudaDeviceSynchronize());
}




/** We configure the GPU according to the user configuration:
 * 1) Guess the "best" number of blocks per SM, if not provided.
 * 2) Update the number of subproblems to at least "30 * B" where B is the number of blocks.
 * 3) Configure the size of the shared memory.
 * 4) Increase the global heap memory.
 * 5) Increase the stack size if requested by the user.
 */
MemoryConfig configure_gpu_barebones(CP<Itv>& cp) {
  auto& config = cp.config;

  /** I. Number of blocks per SM. */
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int max_block_per_sm;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_block_per_sm, (void*) gpu_barebones_solve, CUDA_THREADS_PER_BLOCK, 0);
  if(cp.config.verbose_solving) {
    printf("%% max_blocks_per_sm=%d\n", max_block_per_sm);
  }
  if(cp.config.or_nodes != 0) {
    cp.stats.num_blocks = std::min(max_block_per_sm * deviceProp.multiProcessorCount, (int)cp.config.or_nodes);
    if(cp.config.verbose_solving >= 1 && cp.stats.num_blocks < cp.config.or_nodes) {
      printf("%% WARNING: -or %d is too high on your GPU architecture, it has been reduced to %d.\n", (int)cp.config.or_nodes, cp.stats.num_blocks);
    }
  }
  else {
    cp.stats.num_blocks = max_block_per_sm * deviceProp.multiProcessorCount;
  }

  /** II. Number of subproblems, if not set, calculate the minimum */
  cp.stats.print_stat("subproblems_power", cp.config.subproblems_power);
  if(cp.config.subproblems_power == -1) {
    cp.config.subproblems_power = 0;
    while((1 << cp.config.subproblems_power) < cp.config.subproblems_factor * cp.stats.num_blocks) {
      cp.config.subproblems_power++;
    }
  }

  /** III. Size of the heap global memory.
   * The estimation is very conservative, normally we should not run out of memory.
   * */
  //the number of bytes to keep every interval
  size_t store_bytes = gpu_sizeof<IStore>() + gpu_sizeof<abstract_ptr<IStore>>() + cp.store->vars() * gpu_sizeof<Itv>();
  //the number of bytes to keep every constraint
  size_t iprop_bytes = gpu_sizeof<IProp>() + gpu_sizeof<abstract_ptr<IProp>>() + cp.iprop->num_deductions() * gpu_sizeof<bytecode_type>() + gpu_sizeof<typename IProp::bytecodes_type>();
  
  //how much a block needs to store its data (store, constraints, strategies, etc.)
  size_t mem_per_block = gpu_sizeof<BlockData>()
    + store_bytes * size_t{3}  // current, root, best.
    + store_bytes * size_t{2}  // search strategies
    + iprop_bytes * size_t{2}
    + cp.iprop->num_deductions() * size_t{4} * gpu_sizeof<int>()  // fixpoint engine
    + (gpu_sizeof<int>() + gpu_sizeof<LightBranch<Itv>>()) * size_t{MAX_SEARCH_DEPTH};
  
  if(config.enr_restart_on()) { //ENR CODE --begin : snapshot (store + stack copy) and the item quota of the block.
    mem_per_block += store_bytes + gpu_sizeof<LightBranch<Itv>>() * size_t{MAX_SEARCH_DEPTH} + gpu_sizeof<EnrFrontierItem>() * size_t{1024}; //ENR CODE
  } //ENR CODE --end

  size_t estimated_global_mem = gpu_sizeof<UnifiedData>() + store_bytes * size_t{5} + iprop_bytes +
    gpu_sizeof<GridData>();

  size_t mem_for_blocks = deviceProp.totalGlobalMem - estimated_global_mem - (deviceProp.totalGlobalMem / 100 * 10);
  cp.stats.num_blocks = std::max(size_t{1}, std::min(mem_for_blocks / mem_per_block, static_cast<size_t>(cp.stats.num_blocks)));
  estimated_global_mem += cp.stats.num_blocks * mem_per_block;
  if(estimated_global_mem > deviceProp.totalGlobalMem / 100 * 90) {
    printf("%% WARNING: The estimated global memory is larger than 90%% of the total global memory.\n\
%% It is possible to run out of memory during solving.\n");
  }
  //ENR CODE -- begin: heap sized from the free memory (upstream: share of the total memory).
  /** Reserving 97% of the *total* memory leaves no room for the per-launch local memory (stack) when part of the GPU is already used (e.g. by the display),
   * and every kernel launch then fails with `cudaErrorLaunchOutOfResources`. We size the heap from the memory actually free instead. */
  size_t free_mem, total_mem;
  CUDAEX(cudaMemGetInfo(&free_mem, &total_mem));
  CUDAEX(cudaDeviceSetLimit(cudaLimitMallocHeapSize, free_mem / 100 * 95)); // 95% of the free memory (was 80%)
  //ENR CODE -- end
  cp.stats.print_memory_statistics(cp.config.verbose_solving, "heap_memory", estimated_global_mem);
  cp.stats.print_memory_statistics(cp.config.verbose_solving, "mem_per_block", mem_per_block);
  cp.stats.print_memory_statistics(cp.config.verbose_solving, "total_global_mem_bytes", deviceProp.totalGlobalMem);

  // We still need to improve this, for some large problems, it is required to avoid running out of memory.
  cp.stats.num_blocks = std::min(cp.stats.num_blocks, 200000000 / cp.store->vars());
  cp.stats.print_stat("num_blocks", cp.stats.num_blocks);








  /** IV. Increase the stack if requested by the user. */
  if(config.stack_kb != 0) {
    CUDAEX(cudaDeviceSetLimit(cudaLimitStackSize, config.stack_kb*1000));
    // The stack allocated depends on the maximum number of threads per SM, not on the actual number of threads per block.
    size_t total_stack_size = deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor * config.stack_kb * 1000;
    cp.stats.print_memory_statistics(cp.config.verbose_solving, "stack_memory", total_stack_size);
  }

  /** V. Configure the shared memory size. */
  int blocks_per_sm = std::max(1, (cp.stats.num_blocks + deviceProp.multiProcessorCount - 1) / deviceProp.multiProcessorCount);
  MemoryConfig mem_config;
  if(config.only_global_memory) {
    mem_config = MemoryConfig(store_bytes, iprop_bytes);
  }
  else {
    mem_config = MemoryConfig((void*) gpu_barebones_solve, config.verbose_solving, blocks_per_sm, store_bytes, iprop_bytes);
  }
  mem_config.print_mzn_statistics(config, cp.stats);

  //ENR CODE -- begin: the restarts synchronize the whole grid (grid.sync()), which requires a cooperative launch, check if supported
  // where all the blocks are resident at the same time.
  if(config.enr_restart_on()) {
    int coop = 0;
    CUDAEX(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0));
    int bps = 0;
    CUDAEX(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, (void*) gpu_barebones_solve, CUDA_THREADS_PER_BLOCK, mem_config.shared_bytes));
    int max_resident = bps * deviceProp.multiProcessorCount;
    if(!coop || max_resident < 1) {
      printf("%% WARNING: -restart requires a cooperative launch, not supported here: restarts are disabled.\n");
      config.enr_restart_idle_pct = 0;
      config.enr_restart_each_block = false;
    }
    //shoud never fire by design
    else if(cp.stats.num_blocks > max_resident) {
      printf("%% WARNING: -restart: %d blocks cannot be resident at the same time, reduced to %d.\n", cp.stats.num_blocks, max_resident);
      cp.stats.num_blocks = max_resident;
      cp.stats.print_stat("num_blocks", cp.stats.num_blocks);
    }
  }
  //ENR CODE -- end
  return mem_config;
}

__global__ void initialize_global_data(
  UnifiedData* unified_data,
  bt::unique_ptr<GridData, bt::global_allocator>* grid_data_ptr)
{
  *grid_data_ptr = bt::make_unique<GridData, bt::global_allocator>(unified_data->root);
}

#define TIMEPOINT(KIND) \
  if(threadIdx.x == 0) { \
    block_data.timer = block_data.stats.stop_timer(Timer::KIND, block_data.timer); \
  }

//ENR STATS: charge the time since the previous ENR_PH to phase P (thread 0 only); CNT is added to the phase counter.
// NOW must be a time point valid on thread 0 (block_data.timer right after a TIMEPOINT, or a fresh clock read).
#define ENR_PH(P, NOW, CNT) \
  if(threadIdx.x == 0) { \
    auto enr_now_ = (NOW); \
    block_data.stats.enr_ph_ns[Statistics<>::P] += cuda::std::chrono::duration_cast<cuda::std::chrono::nanoseconds>(enr_now_ - block_data.enr_ph_t).count(); \
    block_data.stats.enr_ph_cnt[Statistics<>::P] += (CNT); \
    block_data.enr_ph_t = enr_now_; \
  }











__global__ void gpu_barebones_solve(UnifiedData* unified_data, GridData* grid_data) {

  extern __shared__ unsigned char shared_mem[];
  auto& config = unified_data->root.config;
  //get my block of data
  BlockData& block_data = grid_data->blocks[blockIdx.x];

  if(threadIdx.x == 0 && blockIdx.x == 0 && config.verbose_solving) {
    printf("%% GPU kernel started, starting solving...\n");
  }
  //ENR STATS: always printed once, proves this local build is the one running and which lala-core it was compiled with.
  if(threadIdx.x == 0 && blockIdx.x == 0) {
    #ifdef LALA_CORE_LOCAL_ENR
        printf("%% [ENR] local Turbo build running (search kernel), lala-core = LOCAL clone ~/Scrivania/lala-core\n");
    #else
        printf("%% [ENR] local Turbo build running (search kernel), lala-core = downloaded copy in build/_deps\n");
    #endif
  }

  /** A. Initialization the block data and the fixpoint engine. */
  block_data.allocate(*unified_data, *grid_data, shared_mem); //ENR: includes also the restart block data if enabled 
  __syncthreads(); //sync block

  // keep a pointer to the block propagators
  IProp& iprop = *block_data.iprop;

  #ifdef TURBO_NO_ENTAILED_PROP_REMOVAL
    __shared__ BlockAsynchronousFixpointGPU<true> fp_engine;
  #else
    __shared__ FixpointSubsetGPU<BlockAsynchronousFixpointGPU<true>, bt::global_allocator, CUDA_THREADS_PER_BLOCK> fp_engine;
    fp_engine.init(iprop.num_deductions());
  #endif



  /** This shared variable is necessary to avoid multiple threads to read into `unified_data.stop.test()`,
   * potentially reading different values and leading to deadlock. */
  __shared__ bool stop;
  __shared__ bool has_changed;
  __shared__ bool is_leaf_node;
  __shared__ int remaining_depth;
  stop = false;

  //just a way to say block
  auto group = cooperative_groups::this_thread_block();

  
  //ENR CODE -- begin: round loop. Round 0 is the EPS search, every restart starts a new round on the open nodes
  // collected from the busy blocks. Without -restart the round loop runs once and the kernel behaves as before.
  __shared__ bool enr_restart;             // shared copy of grid_data->enr_restart_flag, set by thread 0 in propagate(), as the stop flag
  __shared__ bool enr_open_subproblem;     // this block stopped because of the restart and its subproblem is not finished.
  __shared__ int enr_s_bits;               // split depth of the round: index idx = item (idx >> enr_s_bits) + dive bits (low enr_s_bits bits).
  __shared__ size_t num_subproblems;   // number of indices of the round (round 0: 2^subproblems_power, then n * 2^s).
  __shared__ EnrFrontierItem enr_cur_item;    // the item being rebuilt (step C').

  if(threadIdx.x == 0) {
    block_data.timer = block_data.stats.start_timer_device();
    block_data.start_time = block_data.timer;
    block_data.enr_ph_t = block_data.timer; //ENR STATS

    //ENR CODE -- begin: at the start of each round, set the flags
    enr_restart = false;
    enr_open_subproblem = false;
    enr_s_bits = config.subproblems_power;
    num_subproblems = unified_data->root.stats.eps_num_subproblems;
    if(blockIdx.x == 0) { grid_data->enr_kernel_start_ns = enr_restart_now_ns(); } //ENR STATS
    //ENR CODE -- end
  }
  __syncthreads(); 

  /** B. Start the main dive and solve loop. */

  int enr_round = 0; // number of barriers passed (S1..S3), the same value in every block.
  //ENR CODE (restart rounds)
  while(true) {
  
    //keep looping until all solved or stop
    while(block_data.subproblem_idx < num_subproblems && !stop) {
      
      if(config.verbose_solving >= 2 && threadIdx.x == 0) {
        grid_data->print_lock.acquire();
        printf("%% Block %d solves subproblem num %" PRIu64 "\n", blockIdx.x, block_data.subproblem_idx);
        grid_data->print_lock.release();
      }

      // C. Restoring the current state to the root node.

      block_data.current_strategy = 0;
      block_data.next_unassigned_var = 0;
      block_data.depth = 0;

      if(enr_round == 0) { //ENR CODE --begin
        // copy just the variables at the subproblem root from the global root (depth 0) as if no restarts
        unified_data->root.store->copy_to(group, *block_data.store);
      }
      //C' Rebuild the open node (item) idx >> enr_s_bits from the snapshot of the block that left it:
      // copy its root store, re-apply its path down to level k, then take child c at level k (as a backtrack does, step IV).
      else {
        if(threadIdx.x == 0) {
          enr_cur_item = grid_data->enr_items[block_data.subproblem_idx >> enr_s_bits];
        }
        __syncthreads();
        const BlockData& src = grid_data->blocks[enr_cur_item.snap];
        src.enr_snap_store->copy_to(group, *block_data.store);
        if(threadIdx.x == 0) {
          has_changed = true;
        }
        __syncthreads();
        int item_k = static_cast<int>(enr_cur_item.k);
        while(has_changed) {
          __syncthreads();
          if(threadIdx.x == 0) {
            has_changed = false;
          }
          __syncthreads();
          for(int i = threadIdx.x; i < item_k; i += blockDim.x) {
            if(block_data.store->embed(src.enr_snap_dec[i].var, src.enr_snap_dec[i].current())) {
              has_changed = true;
            }
          }
          __syncthreads();
        }
        if(threadIdx.x == 0) {
          block_data.store->embed(src.enr_snap_dec[item_k].var, src.enr_snap_dec[item_k].children[enr_cur_item.c]);
          block_data.current_strategy = src.enr_snap_root_strategy; // next_unassigned_var stays 0 (step C): the split skips the assigned variables.
        }
      }
      //ENR CODE -- end

      #ifndef TURBO_NO_ENTAILED_PROP_REMOVAL
          fp_engine.reset(iprop.num_deductions());
      #endif


      //ENR needed(?)
      __syncthreads();

      // D. Dive into the search tree until we reach the target subproblem.
      remaining_depth = enr_s_bits; //ENR CODE: was config.subproblems_power
      if(threadIdx.x == 0) {
        block_data.dive_timer = block_data.stats.start_timer_device();
        is_leaf_node = false;
      }
      ENR_PH(PH_SUBPB_SWITCH, block_data.dive_timer, 1); //ENR STATS: G + loop head + C (copy of the global root) of this subproblem; counter = number of dives.

      __syncthreads();

      //dive (still quite fast)

      while(remaining_depth > 0 && !is_leaf_node && !stop) {
        //needed(?)
        __syncthreads();
        
        ENR_PH(PH_DIVE_BRANCH, block_data.stats.start_timer_device(), 0); //ENR STATS: split + commit of the previous dive level
        // This dive node is shared by all the indices with the same high bits: only the smallest one (low bits = 0) counts it. //ENR STATS
        if(threadIdx.x == 0) { block_data.enr_in_tree = (block_data.subproblem_idx & ((size_t{1} << remaining_depth) - size_t{1})) == size_t{0}; } //ENR STATS
        propagate(*unified_data, *grid_data, block_data, fp_engine, stop, enr_restart, has_changed, is_leaf_node); //ENR CODE: + enr_restart
        __syncthreads();
        ENR_PH(PH_DIVE_FP, block_data.stats.start_timer_device(), 1); //ENR STATS: the whole propagate() of a dive node
        if(!is_leaf_node) {
          block_data.split(has_changed, grid_data->search_strategies);
          __syncthreads();
          // Split was not able to split a domain. It means that the search strategy is not complete due to unsplittable infinite domains.
          // We skip the subtree, and set exhaustive to `false`.
          if(block_data.decisions[0].var.is_untyped()) {
            is_leaf_node = true;
            block_data.stats.exhaustive = false;
            if(threadIdx.x == 0 && config.verbose_solving >= 1) { printf("%% WARNING: infinite element detected during branching, search is not exhaustive\n");}
          }
          else if(threadIdx.x == 0) {
            --remaining_depth;
            // We do not record the decisions when diving.
            --block_data.depth;
            /** We commit to one of the branches depending on the current value on the path.
             * Suppose the depth is 3, the path is "010" we are currently at `remaining_depth = 1`.
             * We must extract the bit "1", and we do so by standard bitwise manipulation.
             * Whenever the branch_idx is 0 means to take the left branch, and 1 means to take the right branch.
             */
            size_t branch_idx = (block_data.subproblem_idx & (size_t{1} << remaining_depth)) >> remaining_depth;
            /** We immediately commit to the branch. */
            // printf("split on %d (", block_data.decisions[0].var.vid()); block_data.store->project(block_data.decisions[0].var).print(); printf(")\n");
            block_data.store->embed(block_data.decisions[0].var, block_data.decisions[0].children[branch_idx]);
          }
        }
        __syncthreads();
      }




      if(threadIdx.x == 0) {
        auto enr_dive_end = block_data.stats.stop_timer(Timer::DIVE, block_data.dive_timer); //ENR STATS: was a plain call
        ENR_PH(PH_DIVE_BRANCH, enr_dive_end, 0); //ENR STATS: split + commit of the last dive level
      }

      

      // E. Skip subproblems that are not reachable.

      /** If we reached a leaf node before the subproblem was reached, then it means a whole subtree should be skipped. */
      if(is_leaf_node && !stop) {
        /** To skip all the paths of the subtree obtained, we perform bitwise operations.
         * Suppose the current path is "00" turn left two times, and the following search tree:
         *         *         depth = 0
         *        / \
         *      0    1       depth = 1
         *    / \   / \
         *   00 01 10 11     depth = 2
         *
         * If we detect a leaf node at depth 1, after only one left turn, we must skip the remaining of the subtree, in particular to avoid exploring the path "01".
         * To achieve that, we take the current path "00", shift it to the right by 1 (essentially erasing the path that has not been explored), increment it to go to the next subtree (at the same depth), and shift it back to the left to reach the first subproblem of the subtree.
         * Cool huh?
         */
        if(threadIdx.x == 0) {
          size_t next_subproblem_idx = ((block_data.subproblem_idx >> remaining_depth) + size_t{1}) << remaining_depth;
          // Make sure the subtree is skipped.
          //ENR CODE -- begin (bug fix): `meet` is load/compare/store, not an atomic max: between the load and the store other blocks can
          // increment the counter past next_subproblem_idx, and the store would move it BACK, handing out the same indices twice.
          // while(grid_data->next_subproblem.meet(ZLB<size_t, bt::local_memory>(next_subproblem_idx))) {}
          grid_data->next_subproblem.atomic().fetch_max(next_subproblem_idx, cuda::memory_order_relaxed);
          //ENR CODE -- end
          /** It is possible that other blocks skip similar subtrees.
            * Hence, we only count the subproblems skipped by the block solving the left most subproblem. */
          if((block_data.subproblem_idx & ((size_t{1} << remaining_depth) - size_t{1})) == size_t{0}) {
            block_data.stats.eps_skipped_subproblems += next_subproblem_idx - block_data.subproblem_idx;
          }
        }
      }
      else if(!stop) {

        // F. Solve the current subproblem.

        // We skip the remaining of the EPS strategy if there is any.
        if(threadIdx.x == 0 && grid_data->has_eps_strategy) {
          block_data.current_strategy = battery::max(1, block_data.current_strategy);
          block_data.next_unassigned_var = 0;
        }
        if(threadIdx.x == 0) { block_data.enr_in_tree = true; } //ENR STATS: every node of the DFS is visited once.

        while(!stop) {

          // I. Optimize the objective variable (only if not diving).

          if(threadIdx.x == 0 && !grid_data->obj_var.is_untyped()) {
            /** Before propagating, we update the local bound with the best known global bound.
             * Strenghten the objective variable to get a better objective next time.
             */
            if(!grid_data->appx_best_bound.is_top()) {
              block_data.store->embed(grid_data->obj_var,
                Itv(Itv::LB::top(), Itv::UB(grid_data->appx_best_bound.value() - 1)));
              block_data.store->embed(grid_data->obj_var,
                Itv(Itv::LB::top(), Itv::UB(block_data.best_bound.value() - 1)));
            }
            // Unconstrained objective, can terminate, we will not find a better solution.
            if(grid_data->appx_best_bound.is_bot()) {
              stop = true;
              unified_data->stop.test_and_set();
            }
          }
          __syncthreads();
          if(stop) {
            break;
          }

          // II. Propagate the current node.
          bool enr_bt = block_data.enr_after_restore; //ENR STATS: read before propagate() clears it; only thread 0's value is used
          ENR_PH(PH_SEARCH_BRANCH, block_data.stats.start_timer_device(), 0); //ENR STATS: objective bound, snapshot, split, decision since the previous node
          propagate(*unified_data, *grid_data, block_data, fp_engine, stop, enr_restart, has_changed, is_leaf_node); //ENR CODE: + enr_restart
          __syncthreads();
          if(enr_bt) { ENR_PH(PH_BT_FP, block_data.stats.start_timer_device(), 1); } //ENR STATS: the whole propagate() of the node reached by a backtrack
          else { ENR_PH(PH_SEARCH_FP, block_data.stats.start_timer_device(), 1); } //ENR STATS: the whole propagate() of a node reached by going down

          // III. Branching

          if(!is_leaf_node) {
            // If we are at the root of the current subproblem, we create a snapshot for future backtracking.
            if(block_data.depth == 0) {
              block_data.store->copy_to(group, *block_data.root_store);
              if(threadIdx.x == 0) {
                block_data.snapshot_root_strategy = block_data.current_strategy;
                block_data.snapshot_next_unassigned_var = block_data.next_unassigned_var;
              }
            }
            __syncthreads();
            block_data.split(has_changed, grid_data->search_strategies);
            __syncthreads();
            // Split was not able to split a domain. It means that the search strategy is not complete due to unsplittable infinite domains.
            // We trigger backtracking, and set exhaustive to `false`.
            if(block_data.decisions[block_data.depth - 1].var.is_untyped()) {
              is_leaf_node = true;
              block_data.stats.exhaustive = false;
              if(threadIdx.x == 0 && config.verbose_solving >= 1) { printf("%% WARNING: infinite element detected during branching, search is not exhaustive\n");}
            }
            else if(threadIdx.x == 0) {
              // Apply the decision.
              // printf("split on %d (", block_data.decisions[block_data.depth-1].var.vid()); block_data.store->project(block_data.decisions[block_data.depth-1].var).print(); printf(")\n");
              block_data.store->embed(block_data.decisions[block_data.depth-1].var, block_data.decisions[block_data.depth-1].next());
              // printf("left decision: %d [", block_data.decisions[block_data.depth - 1].var.vid()); block_data.decisions[block_data.depth - 1].current().print(); printf("]\n");
            }
          }

          // IV. Backtracking

          if(is_leaf_node) {
            // Leaf node at root.
            if(block_data.depth == 0) {
              break;
            }
            if(threadIdx.x == 0) {
              block_data.depth = block_data.decisions[block_data.depth-1].ropes[block_data.decisions[block_data.depth-1].current_idx];
            }
            __syncthreads();
            // Check if there is no more node to visit.
            if(block_data.depth == -1) {
              break;
            }
            // Restore from root by copying the store and re-applying all decisions from root to block_data.depth-1.
            TIMEPOINT(SEARCH); //ENR STATS: close the split/other interval before timing the restore.
            ENR_PH(PH_SEARCH_BRANCH, block_data.timer, 0); //ENR STATS: split/decision of the previous node is already charged; here: rope lookup of the failed node
            if(threadIdx.x == 0) { //ENR STATS
              block_data.stats.enr_backtracks++; //ENR STATS
              block_data.enr_restore_start = block_data.timer; //ENR STATS
            } //ENR STATS
            #ifndef TURBO_NO_ENTAILED_PROP_REMOVAL
                      fp_engine.reset(iprop.num_deductions());
            #endif
            block_data.root_store->copy_to(group, *block_data.store);
            // __syncthreads();
            // if(threadIdx.x == 0) {
            //   printf("%d: restoring store: ", block_data.depth); block_data.store->print(); printf("\n");
            // }
            // __syncthreads();
            if(threadIdx.x == 0) {
              has_changed = true;
            }
            __syncthreads();
            ENR_PH(PH_BT_RESTORE, block_data.stats.start_timer_device(), 1); //ENR STATS: copy root_store -> store (all threads done)
            while(has_changed) {
              __syncthreads();
              if(threadIdx.x == 0) {
                has_changed = false;
              }
              __syncthreads();
              for(int i = threadIdx.x; i < block_data.depth - 1; i += blockDim.x) {
                if(block_data.store->embed(block_data.decisions[i].var, block_data.decisions[i].current())) {
                  has_changed = true;
                }
              }
              __syncthreads();
            }
            if(threadIdx.x == 0) {
              block_data.store->embed(block_data.decisions[block_data.depth - 1].var, block_data.decisions[block_data.depth - 1].next());
              // printf("right decision: %d [", block_data.decisions[block_data.depth - 1].var.vid()); block_data.decisions[block_data.depth - 1].current().print(); printf("]\n");
              block_data.current_strategy = block_data.snapshot_root_strategy;
              block_data.next_unassigned_var = block_data.snapshot_next_unassigned_var;
            }
            TIMEPOINT(BACKTRACK_ENR); //ENR STATS
            ENR_PH(PH_BT_REDESCEND, block_data.timer, 1); //ENR STATS: re-apply the decisions of the path + take the right branch
            if(threadIdx.x == 0) { //ENR STATS
              block_data.enr_restore_ns = cuda::std::chrono::duration_cast<cuda::std::chrono::nanoseconds>(block_data.timer - block_data.enr_restore_start).count(); //ENR STATS
              block_data.enr_restore_path = block_data.depth - 1; //ENR STATS
              block_data.enr_after_restore = true; //ENR STATS
            } //ENR STATS
            // __syncthreads();
            // if(threadIdx.x == 0) {
            //   printf("%d: reapplied decisions: ", block_data.depth); block_data.store->print(); printf("\n");
            // }
            // __syncthreads();
          }

          //ENR CODE -- begin: checked at the END of the body, so `root_store` has been taken (depth 0) and the last split
          // or backtrack is embedded but not propagated: the current node is unexplored, nothing is lost or done twice.
          if(enr_restart) {
            if(threadIdx.x == 0) {
              enr_open_subproblem = !stop;
            }
            __syncthreads();
            break;
          }
          //ENR CODE -- end
        }
        /** If we didn't stop solving because of an external interruption, we increase the number of subproblems solved. */
        if(threadIdx.x == 0 && block_data.stats.nodes < config.stop_after_n_nodes
          && !unified_data->stop.test() && !enr_open_subproblem) //ENR CODE: && !enr_open_subproblem
        {
          block_data.stats.eps_solved_subproblems += 1;
        }
      }

      // G. Move to the next subproblem.

      /** We prepare the block to solve the next problem.
       * We update the subproblem index to the next subproblem to solve. */
      if(threadIdx.x == 0 && !stop) {
        /** To avoid that several blocks solve the same subproblem, we use an atomic post-increment. */
        block_data.subproblem_idx = grid_data->next_subproblem.atomic()++;
        /** The following commented code is completely valid and does not use atomic post-increment.
         * But honestly, we kinda need more performance so... let's avoid reexploring subproblems. */
        // subproblem_idx = grid_data->next_subproblem.value();
        // grid_data->next_subproblem.meet(ZLB<size_t, bt::local_memory>(subproblem_idx + size_t{1}));
      }
      __syncthreads();
    }

    //ENR CODE -- begin: end of a round.
    // Without -restart, leave now (same value in every block): the kernel ends exactly as before.
    if(!grid_data->enr_restart_enabled) {
      break;
    }
    {
      auto grid = cooperative_groups::this_grid();
      if(threadIdx.x == 0) {
        // A block that left the loop without stop and without restart ran out of work: it is idle.
        // The idle block that brings the count to the threshold raises the flag (exactly one block, fetch_add values are unique);
        // the busy blocks see it in propagate() and stop at the end of their current node.
        if(!stop && !enr_restart) {
          int idle = grid_data->enr_idle_blocks.fetch_add(1, cuda::memory_order_relaxed) + 1;
          if(idle == grid_data->enr_restart_threshold) {
            grid_data->enr_restart_flag.store(1, cuda::memory_order_relaxed);
            grid_data->enr_raise_ns = enr_restart_now_ns(); //ENR STATS
          }
        }
        auto now = cuda::std::chrono::system_clock::now(); //ENR STATS
        ENR_PH(PH_SUBPB_SWITCH, now, 0); //ENR STATS: end of the round -> barrier (the wait is idle time)
        block_data.enr_wait_begin = cuda::std::chrono::duration_cast<cuda::std::chrono::nanoseconds>(now - block_data.start_time).count(); //ENR STATS
      }

      grid.sync(); // S1: nobody explores; every stack, root_store and best_bound is frozen; nobody reads the snapshots, the counter or the flag.
      enr_round++;

      // Harvest: a block stopped by the restart publishes its open nodes. It writes only its own snapshot,
      // its own range of `items` (reserved with one atomicAdd) and the counter enr_n_items[round & 1].
      if(enr_open_subproblem) {
        if(threadIdx.x == 0 && block_data.enr_snap_dec.size() < block_data.decisions.size()) {
          block_data.enr_snap_dec.resize(block_data.decisions.size()); // `decisions` may have grown (push_decision doubles it).
        }
        __syncthreads();
        block_data.root_store->copy_to(group, *block_data.enr_snap_store);
        for(int i = threadIdx.x; i < block_data.depth; i += blockDim.x) {
          LightBranch<Itv>& d = block_data.enr_snap_dec[i];
          const LightBranch<Itv>& e = block_data.decisions[i];
          d.var = e.var;
          d.children[0] = e.children[0];
          d.children[1] = e.children[1];
          d.current_idx = e.current_idx;
        }
        if(threadIdx.x == 0) {
          int depth = block_data.depth;
          block_data.enr_snap_root_strategy = block_data.snapshot_root_strategy;
          if(depth >= 1) {
            // Open nodes: (k, 1) for every level whose right branch was never started (current_idx == 0),
            // plus the current node (depth-1, current_idx of the top level). No left branch other than the current one can be open.
            int pending = 0;
            for(int k = 0; k < depth; ++k) {
              if(block_data.decisions[k].current_idx == 0) { ++pending; }
            }
            int keep = pending;   // pending right branches published.
            int last_k = depth - 1; // the last item is (last_k, current_idx of last_k).
            if(pending + 1 > grid_data->enr_items_per_block) {
              // Quota exceeded (never expected): keep the shallowest pending right branches and cover everything deeper
              // by the left child of the last kept level (it re-explores some refuted parts, but nothing is lost).
              keep = grid_data->enr_items_per_block - 1;
              int seen = 0;
              for(int k = 0; k < depth; ++k) {
                if(block_data.decisions[k].current_idx == 0 && ++seen == keep) { last_k = k; break; }
              }
              grid_data->enr_st_truncated.fetch_add(1, cuda::memory_order_relaxed); //ENR STATS
            }
            int m = keep + 1;
            int w = grid_data->enr_n_items[enr_round & 1].fetch_add(m, cuda::memory_order_relaxed);
            int seen = 0;
            for(int k = 0; k < depth && seen < keep; ++k) {
              if(block_data.decisions[k].current_idx == 0) {
                grid_data->enr_items[w].snap = blockIdx.x;
                grid_data->enr_items[w].k = k;
                grid_data->enr_items[w].c = 1;
                ++w; ++seen;
              }
            }
            grid_data->enr_items[w].snap = blockIdx.x;
            grid_data->enr_items[w].k = last_k;
            grid_data->enr_items[w].c = block_data.decisions[last_k].current_idx;
          }
        }
      }

      // reset_round: one writer (block 0) for every shared field of the next round. Nothing it writes is read
      // between S1 and S3, and nothing the harvests write is read by it, so it runs in parallel with the harvests.
      if(blockIdx.x == 0 && threadIdx.x == 0) {
        int64_t now = enr_restart_now_ns(); //ENR STATS
        // (1) exact best bound: every block is stopped, the minimum is exact (repairs the racy meet of propagate()).
        for(int b = 0; b < grid_data->blocks.size(); ++b) {
          grid_data->appx_best_bound.meet(grid_data->blocks[b].best_bound);
        }
        // (2) the host flag, read once: every block sees the same value after S3.
        grid_data->enr_host_stop = unified_data->stop.test();
        //ENR STATS -- begin: drain time of this barrier (flag raised -> all blocks at S1).
        if(grid_data->enr_restart_flag.load(cuda::memory_order_relaxed) != 0) {
          grid_data->enr_pending_drain_ns = now - grid_data->enr_raise_ns;
          grid_data->enr_pending_raise_ns = grid_data->enr_raise_ns;
        }
        else {
          grid_data->enr_pending_drain_ns = -1;
        }
        //ENR STATS -- end
        // (3) counter: blocks 0..B-1 start on their own index, the counter hands out B onward (a store, meet() only increases).
        grid_data->next_subproblem = ZLB<size_t, bt::local_memory>(grid_data->blocks.size());
        // (4)(5)
        grid_data->enr_idle_blocks.store(0, cuda::memory_order_relaxed);
        grid_data->enr_restart_flag.store(0, cuda::memory_order_relaxed);
        // (6) the counter the NEXT restart will fill (the current one is being filled and is read after S3).
        grid_data->enr_n_items[(enr_round + 1) & 1].store(0, cuda::memory_order_relaxed);
      }

      grid.sync(); // S3: items, enr_n_items and the reset are visible to every block.

      size_t n = static_cast<size_t>(grid_data->enr_n_items[enr_round & 1].load(cuda::memory_order_relaxed));
      bool finished = grid_data->enr_host_stop || n == 0; // same value in every block.
      if(threadIdx.x == 0) {
        auto now = cuda::std::chrono::system_clock::now(); //ENR STATS
        int64_t wait_end = cuda::std::chrono::duration_cast<cuda::std::chrono::nanoseconds>(now - block_data.start_time).count(); //ENR STATS
        block_data.enr_wait_ns += wait_end - block_data.enr_wait_begin; //ENR STATS
        if(block_data.enr_n_waits < BlockData::ENR_MAX_WAITS) { //ENR STATS
          block_data.enr_wait_from[block_data.enr_n_waits] = block_data.enr_wait_begin; //ENR STATS
          block_data.enr_wait_to[block_data.enr_n_waits] = wait_end; //ENR STATS
          block_data.enr_n_waits++; //ENR STATS
        } //ENR STATS
        block_data.enr_ph_t = now; //ENR STATS: the wait is not charged to any phase.
        block_data.timer = now;    //ENR STATS: nor to the timers (TIMEPOINT).
        block_data.enr_after_restore = false; //ENR STATS: a backtrack interrupted by the restart is not followed by its fixpoint.
        if(!finished) {
          // Split depth: smallest s with n * 2^s >= T, so the round has at least T subproblems.
          int s = 0;
          while((n << s) < grid_data->enr_restart_target && s < 40) { ++s; }
          enr_s_bits = s;
          num_subproblems = n << s;
          enr_restart = false;
          enr_open_subproblem = false;
          block_data.subproblem_idx = blockIdx.x;
          if(blockIdx.x == 0) {
            //ENR STATS -- begin
            grid_data->enr_st_restarts++;
            grid_data->enr_st_items_last = n;
            grid_data->enr_st_items_sum += n;
            grid_data->enr_st_items_max = battery::max(grid_data->enr_st_items_max, n);
            grid_data->enr_st_subproblems_sum += num_subproblems;
            grid_data->enr_st_s_max = battery::max(grid_data->enr_st_s_max, s);
            if(grid_data->enr_pending_drain_ns >= 0) {
              grid_data->enr_st_drain_sum += grid_data->enr_pending_drain_ns;
              grid_data->enr_st_drain_max = battery::max(grid_data->enr_st_drain_max, grid_data->enr_pending_drain_ns);
              int64_t t = grid_data->enr_pending_raise_ns - grid_data->enr_kernel_start_ns;
              if(grid_data->enr_st_first_ns < 0) { grid_data->enr_st_first_ns = t; }
              grid_data->enr_st_last_ns = t;
            }
            //ENR STATS -- end
          }
        }
      }
      __syncthreads();
      if(finished) {
        break;
      }
    }
  } // while(true): next round
  //ENR CODE -- end

  if(threadIdx.x == 0)
  {
    if(block_data.stats.nodes < config.stop_after_n_nodes && !unified_data->stop.test()) {
      block_data.stats.num_blocks_done = 1;
    }
    ENR_PH(PH_SUBPB_SWITCH, block_data.stats.start_timer_device(), 0); //ENR STATS: end of the last subproblem -> exit
    block_data.stats.timers.update_timer(Timer::FIRST_BLOCK_IDLE, block_data.start_time);
    block_data.stats.cumulative_time_block = block_data.stats.timers.time_of(Timer::FIRST_BLOCK_IDLE);
  }
  __syncthreads();
  #ifndef TURBO_NO_ENTAILED_PROP_REMOVAL
    fp_engine.destroy();
  #endif
  block_data.deallocate_shared_data();
  __syncthreads();
}
























template <class FPEngine>
__device__ INLINE void propagate(UnifiedData& unified_data, GridData& grid_data, BlockData& block_data,
   FPEngine& fp_engine, bool& stop, bool& enr_restart, bool& has_changed, bool& is_leaf_node) //ENR CODE: + bool& enr_restart
{
  __shared__ int warp_iterations[CUDA_THREADS_PER_BLOCK/32];
  warp_iterations[threadIdx.x / 32] = 0;
  auto& config = unified_data.root.config;
  IProp& iprop = *block_data.iprop;
  auto group = cooperative_groups::this_thread_block();

  TIMEPOINT(SEARCH);
  auto enr_fp_start = block_data.timer; //ENR STATS (only meaningful on thread 0)
  if(threadIdx.x == 0) {
    is_leaf_node = false;
  }

  // II. Compute the fixpoint of the current node.
  int fp_iterations;
  #ifdef TURBO_NO_ENTAILED_PROP_REMOVAL
    int num_active = iprop.num_deductions();
  #else
    int num_active = fp_engine.num_active();
  #endif
  switch(config.fixpoint) {
    case FixpointKind::AC1: {
      fp_iterations = fp_engine.fixpoint(
        #ifdef TURBO_NO_ENTAILED_PROP_REMOVAL
                iprop.num_deductions(),
        #endif
        [&](int i){ return iprop.deduce(i); },
        [&](){ return iprop.is_bot(); });
      if(threadIdx.x == 0) {
        block_data.stats.num_deductions += fp_iterations * num_active;
      }
      break;
    }
    case FixpointKind::WAC1: {
      if(num_active <= config.wac1_threshold) {
        fp_iterations = fp_engine.fixpoint(
          #ifdef TURBO_NO_ENTAILED_PROP_REMOVAL
                  iprop.num_deductions(),
          #endif
          [&](int i){ return iprop.deduce(i); },
          [&](){ return iprop.is_bot(); });
        if(threadIdx.x == 0) {
          block_data.stats.num_deductions += fp_iterations * num_active;
        }
      }
      else {
        fp_iterations = fp_engine.fixpoint(
          #ifdef TURBO_NO_ENTAILED_PROP_REMOVAL
                    iprop.num_deductions(),
          #endif
          [&](int i){ return warp_fixpoint<CUDA_THREADS_PER_BLOCK>(iprop, i, warp_iterations); },
          [&](){ return iprop.is_bot(); });
        if(threadIdx.x == 0) {
          for(int i = 0; i < CUDA_THREADS_PER_BLOCK/32; ++i) {
            block_data.stats.num_deductions += warp_iterations[i] * 32;
          }
        }
      }
      break;
    }
  }
  TIMEPOINT(FIXPOINT);
  if(threadIdx.x == 0) { //ENR STATS
    int64_t enr_fp_ns = cuda::std::chrono::duration_cast<cuda::std::chrono::nanoseconds>(block_data.timer - enr_fp_start).count(); //ENR STATS
    block_data.stats.enr_record_node_fp(enr_fp_ns); //ENR STATS
    if(block_data.enr_after_restore) { //ENR STATS
      block_data.stats.enr_record_backtrack(block_data.enr_restore_ns, enr_fp_ns, fp_iterations, block_data.enr_restore_path); //ENR STATS
      block_data.enr_after_restore = false; //ENR STATS
    } //ENR STATS
    else { block_data.stats.enr_record_other_node(enr_fp_ns, fp_iterations); } //ENR STATS
  } //ENR STATS

  // III. Analyze the result of propagation

  if(!iprop.is_bot()) {
    #ifdef TURBO_NO_ENTAILED_PROP_REMOVAL
        if(threadIdx.x == 0) {
          has_changed = false;
        }
        __syncthreads();
        for(int i = threadIdx.x; !has_changed && i < iprop.num_deductions(); i += blockDim.x) {
          if(!iprop.ask(i)) {
            has_changed = true;
          }
        }
        __syncthreads();
        num_active = has_changed ? 1 : 0;
    #else
        fp_engine.select([&](int i) { return !iprop.ask(i); });
        num_active = fp_engine.num_active();
    #endif
    TIMEPOINT(SELECT_FP_FUNCTIONS);
    /** Whenever we reach a solution node, we must have a bound better than the best bound of the local block.
     * Note that it doesn't mean the best bound of the block must be the best bound of the grid.
     * It is to prevent copying a store with a worst bound into `best_store`.
     */
    if(num_active == 0) {
      is_leaf_node = true;
      if(block_data.best_bound.value() > block_data.store->project(grid_data.obj_var).lb().value()) {
        if(threadIdx.x == 0) {
          block_data.best_bound.meet(Itv::UB(block_data.store->project(grid_data.obj_var).lb().value()));
          grid_data.appx_best_bound.meet(block_data.best_bound);
          block_data.stats.timers.update_timer(Timer::LATEST_BEST_OBJ_FOUND, block_data.start_time);
        }
        block_data.store->copy_to(group, *block_data.best_store);
        if(threadIdx.x == 0) {
          block_data.stats.solutions++;
          if(config.verbose_solving >= 2) {
            grid_data.print_lock.acquire();
            printf("%% objective="); block_data.best_bound.print(); printf("\n");
            grid_data.print_lock.release();
          }
        }
      }
    }
  }
  else {
    is_leaf_node = true;
  }

  if(threadIdx.x == 0) {
    block_data.stats.fixpoint_iterations += fp_iterations;
    block_data.stats.nodes++;
    block_data.stats.fails += (iprop.is_bot() ? 1 : 0);
    if(block_data.enr_in_tree) { //ENR STATS: the node is counted once in the search tree.
      block_data.stats.enr_tree_nodes++; //ENR STATS
      block_data.stats.enr_tree_fails += (iprop.is_bot() ? 1 : 0); //ENR STATS
    } //ENR STATS
    block_data.stats.depth_max = battery::max(block_data.stats.depth_max, block_data.depth);

    // IV. Checking stopping conditions.

    if(block_data.stats.nodes >= config.stop_after_n_nodes
      || unified_data.stop.test())
    {
      block_data.stats.exhaustive = false;
      stop = true;
      // With restarts, every block must leave the round loop together: the node limit stops the whole grid. //ENR CODE
      if(grid_data.enr_restart_enabled) { unified_data.stop.test_and_set(); } //ENR CODE
    }

    //ENR CODE -- begin: poll the restart flag once per node (it is raised by an idle block at the end of its round).
    // A block is idle only when the counter is exhausted, so when the flag goes up all the work of the round has been handed out.
    if(grid_data.enr_restart_enabled && !enr_restart && grid_data.enr_restart_flag.load(cuda::memory_order_relaxed) != 0) {
      enr_restart = true;
    }
    //ENR CODE -- end
  }
}

__global__ void reduce_blocks(UnifiedData* unified_data, GridData* grid_data) {
  auto& root = unified_data->root;
  for(int i = 0; i < grid_data->blocks.size(); ++i) {
    root.stats.meet(grid_data->blocks[i].stats);
    int64_t& grid_first_block_idle = root.stats.timers.time_of(Timer::FIRST_BLOCK_IDLE);
    int64_t block_idle = grid_data->blocks[i].stats.timers.time_of(Timer::FIRST_BLOCK_IDLE);
    if(grid_first_block_idle > block_idle) {
      grid_first_block_idle = block_idle;
    }
  }
  //ENR STATS -- begin: grid-level restart statistics.
  root.stats.enr_restart_mode = grid_data->enr_restart_enabled;
  root.stats.enr_restarts = grid_data->enr_st_restarts;
  root.stats.enr_restart_items_sum = grid_data->enr_st_items_sum;
  root.stats.enr_restart_items_max = grid_data->enr_st_items_max;
  root.stats.enr_restart_items_last = grid_data->enr_st_items_last;
  root.stats.enr_restart_subproblems_sum = grid_data->enr_st_subproblems_sum;
  root.stats.enr_restart_s_max = grid_data->enr_st_s_max;
  root.stats.enr_restart_truncated = grid_data->enr_st_truncated.load();
  root.stats.enr_restart_drain_ns_sum = grid_data->enr_st_drain_sum;
  root.stats.enr_restart_drain_ns_max = grid_data->enr_st_drain_max;
  root.stats.enr_restart_first_ns = grid_data->enr_st_first_ns;
  root.stats.enr_restart_last_ns = grid_data->enr_st_last_ns;
  //ENR STATS -- end
  //ENR STATS -- begin: cross-block load balance.
  if(grid_data->blocks.size() > 0) {
    auto& s0 = grid_data->blocks[0].stats;
    root.stats.enr_nodes_block_min = root.stats.enr_nodes_block_max = s0.nodes;
    root.stats.enr_subpb_block_min = root.stats.enr_subpb_block_max = s0.eps_solved_subproblems;
    root.stats.enr_life_block_min = root.stats.enr_life_block_max = s0.cumulative_time_block;
    for(int i = 1; i < grid_data->blocks.size(); ++i) {
      auto& s = grid_data->blocks[i].stats;
      root.stats.enr_nodes_block_min = battery::min(root.stats.enr_nodes_block_min, s.nodes);
      root.stats.enr_nodes_block_max = battery::max(root.stats.enr_nodes_block_max, s.nodes);
      root.stats.enr_subpb_block_min = battery::min(root.stats.enr_subpb_block_min, s.eps_solved_subproblems);
      root.stats.enr_subpb_block_max = battery::max(root.stats.enr_subpb_block_max, s.eps_solved_subproblems);
      root.stats.enr_life_block_min = battery::min(root.stats.enr_life_block_min, s.cumulative_time_block);
      root.stats.enr_life_block_max = battery::max(root.stats.enr_life_block_max, s.cumulative_time_block);
    }
    root.stats.enr_idle_ns_sum = 0;
    for(int i = 0; i < grid_data->blocks.size(); ++i) {
      root.stats.enr_idle_ns_sum += root.stats.enr_life_block_max - grid_data->blocks[i].stats.cumulative_time_block;
      root.stats.enr_idle_ns_sum += grid_data->blocks[i].enr_wait_ns; //ENR STATS: waits at the restart barriers are idle too.
    }
    // Blocks that left the loop because no subproblem was left (num_blocks_done == 1), not because of the stop. //ENR STATS
    root.stats.enr_done_blocks = 0; root.stats.enr_first_done_ns = -1; root.stats.enr_last_done_ns = -1; //ENR STATS
    for(int i = 0; i < grid_data->blocks.size(); ++i) { //ENR STATS
      auto& s = grid_data->blocks[i].stats; //ENR STATS
      if(s.num_blocks_done == 1) { //ENR STATS
        root.stats.enr_done_blocks++; //ENR STATS
        if(root.stats.enr_first_done_ns < 0 || s.cumulative_time_block < root.stats.enr_first_done_ns) { root.stats.enr_first_done_ns = s.cumulative_time_block; } //ENR STATS
        if(root.stats.enr_last_done_ns < 0 || s.cumulative_time_block > root.stats.enr_last_done_ns) { root.stats.enr_last_done_ns = s.cumulative_time_block; } //ENR STATS
      } //ENR STATS
    } //ENR STATS
    // Idle timeline: the kernel time [0, K) is cut in ENR_TL slices; a block is idle in [its lifetime, K). //ENR STATS
    double K = static_cast<double>(root.stats.enr_life_block_max); //ENR STATS
    int TL = root.stats.ENR_TL; //ENR STATS
    for(int k = 0; k < TL; ++k) { //ENR STATS
      double a = K * k / TL, b = K * (k + 1) / TL; //ENR STATS
      double idle = 0; int busy = 0; //ENR STATS
      for(int i = 0; i < grid_data->blocks.size(); ++i) { //ENR STATS
        double life = static_cast<double>(grid_data->blocks[i].stats.cumulative_time_block); //ENR STATS
        double from = life > a ? life : a; //ENR STATS
        if(from < b) { idle += b - from; } //ENR STATS
        //ENR STATS: the waits at the restart barriers [from, to) are idle; a block waiting at the end of the slice is not busy.
        const BlockData& blk = grid_data->blocks[i]; //ENR STATS
        bool waiting_at_b = false; //ENR STATS
        for(int w = 0; w < blk.enr_n_waits; ++w) { //ENR STATS
          double wf = static_cast<double>(blk.enr_wait_from[w]); //ENR STATS
          double wt = static_cast<double>(blk.enr_wait_to[w]); //ENR STATS
          double lo = wf > a ? wf : a; //ENR STATS
          double hi = wt < b ? wt : b; //ENR STATS
          if(hi > lo) { idle += hi - lo; } //ENR STATS
          if(wf < b && b <= wt) { waiting_at_b = true; } //ENR STATS
        } //ENR STATS
        // a block stopped by the timeout counts as busy until the end. //ENR STATS
        if(!waiting_at_b && (life >= b || grid_data->blocks[i].stats.num_blocks_done != 1)) { busy++; } //ENR STATS: !waiting_at_b
      } //ENR STATS
      root.stats.enr_idle_pct_slice[k] = (b > a) ? 100.0 * idle / ((b - a) * grid_data->blocks.size()) : 0.0; //ENR STATS
      root.stats.enr_busy_end_slice[k] = busy; //ENR STATS
    } //ENR STATS
  }
  //ENR STATS -- end
  int best_block_idx = 0;
  for(int i = 0; i < grid_data->blocks.size(); ++i) {
    auto& block = grid_data->blocks[i];
    if(block.stats.solutions > 0) {
      if(root.bab->is_satisfaction()) {
        block.best_store->extract(*root.best);
        break;
      }
      else {
        bool equal_bound = (grid_data->appx_best_bound == block.best_bound);
        bool is_better = grid_data->appx_best_bound.meet(block.best_bound);
        int64_t& grid_best_time = root.stats.timers.time_of(Timer::LATEST_BEST_OBJ_FOUND);
        int64_t block_best_time = block.stats.timers.time_of(Timer::LATEST_BEST_OBJ_FOUND);
        if(is_better || (equal_bound && block_best_time <= grid_best_time)) {
          grid_best_time = block_best_time;
          best_block_idx = i;
        }
      }
    }
  }
  // If we found a bound, we copy the best store into the unified data.
  if(!grid_data->appx_best_bound.is_top()) {
    grid_data->blocks[best_block_idx].best_store->copy_to(*root.best);
  }
}

__global__ void deallocate_global_data(bt::unique_ptr<GridData, bt::global_allocator>* grid_data) {
  grid_data->reset();
}

#endif // TURBO_IPC_ABSTRACT_DOMAIN
#endif // __CUDACC__

#if defined(TURBO_IPC_ABSTRACT_DOMAIN) || !defined(__CUDACC__)

void barebones_dive_and_solve(const Configuration<battery::standard_allocator>& config) {
#ifdef TURBO_IPC_ABSTRACT_DOMAIN
  std::cerr << "-arch barebones does not support IPC abstract domain." << std::endl;
#else
  std::cerr << "You must use a CUDA compiler (nvcc or clang) to compile Turbo on GPU." << std::endl;
#endif
}

#endif

} // namespace barebones

#endif // TURBO_BAREBONES_DIVE_AND_SOLVE_HPP
