#ifndef CHAKRA_FEEDER_V3_DEPENDANCY_SOLVER_H
#define CHAKRA_FEEDER_V3_DEPENDANCY_SOLVER_H

#include <algorithm>
#include <cstdint>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "common.h"
#include "et_def.pb.h"

namespace Chakra {
namespace FeederV3 {

// Compact per-node adjacency: vector beats unordered_set on memory and
// cache locality for typical workload DAGs where each node has only a
// handful of parents/children (<=5 in llama/gpt training graphs).  Input
// to add_node() is always a set (thus already deduplicated), so no runtime
// dedup is needed; if the caller changes to send duplicates we fall back
// to a safe skip.  Saves roughly half the per-rank DependancyResolver RSS
// on 16-rank llama/in_dc (vs the old unordered_set<NodeId> per node).
using NodeIdList = std::vector<NodeId>;

class _DependancyLayer {
 public:
  _DependancyLayer() = default;
  ~_DependancyLayer() {
    this->child_map_parent.clear();
    this->parent_map_child.clear();
    this->dependancy_free_nodes.clear();
    this->ongoing_nodes.clear();
  }
  /**
   * @brief The node has three possible states in a process of resolving
   * dependancy
   *  1. Pending, which means this node is not processed yet, and might be taken
   * if all its parents released.
   *  2. Taken, which means this node is taken by a process, but still in
   * progress. It shouldnt be taken again by other process.
   *  3. Finished(Not in Graph), which means this node is finished.
   *     The child of it may be released if all its parents is finished.
   *  Finished --add--> Pending --take--> Taken --finish--> Finished
   *  Taken --push_back--> Pending
   */
  void add_node(const NodeId& node, const std::unordered_set<NodeId>& parents);
  void add_node_children(
      const NodeId& node,
      const std::unordered_set<NodeId>& children);
  void take_node(const NodeId& node);
  void finish_node(const NodeId& node);
  void push_back_node(const NodeId& node);
  void resolve_dependancy_free_nodes();

  const std::unordered_set<NodeId>& get_dependancy_free_nodes() const;
  const std::unordered_set<NodeId>& get_ongoing_nodes() const;
  // The adjacency getters now return a vector; only legacy_tests calls these,
  // and they'll still compile (iteration via range-for, .find via std::find).
  const NodeIdList& get_children(NodeId node) const;
  const NodeIdList& get_parents(NodeId node) const;

  // §23 mem opt: release all internal storage to free memory (used when a
  // redundant layer is being dropped at resolve time).  After reset(), the
  // layer is equivalent to a freshly-constructed one.
  void reset();

 private:
  std::unordered_map<NodeId, NodeIdList> child_map_parent;
  std::unordered_map<NodeId, NodeIdList> parent_map_child;
  std::unordered_set<NodeId> dependancy_free_nodes;
  std::unordered_set<NodeId> ongoing_nodes;
  bool dirty = true;
  void _helper_allocate_bucket(NodeId node_id);
  // Insert `val` into `dst` only if not already present.  Fast for short
  // lists; callers hold `parents` as a set so duplicates are rare.
  static void _push_unique(NodeIdList& dst, NodeId val) {
    if (std::find(dst.begin(), dst.end(), val) == dst.end())
      dst.push_back(val);
  }
  std::shared_mutex mutex;
};

class DependancyResolver {
 public:
  DependancyResolver(bool enable_data_deps, bool enable_ctrl_deps)
      : enable_data_deps(enable_data_deps), enable_ctrl_deps(enable_ctrl_deps) {
    if (!enable_data_deps)
      if (!enable_ctrl_deps)
        throw std::runtime_error(
            "Should not create a dependancy resolver that resolves neither data nor control dependancy");
  }
  void add_node(const ChakraNode& node);
  void take_node(const NodeId& node);
  void push_back_node(const NodeId& node);
  void finish_node(const NodeId& node);
  void resolve_dependancy_free_nodes();

  const std::unordered_set<NodeId>& get_dependancy_free_nodes() const;
  const std::unordered_set<NodeId>& get_ongoing_nodes() const;
  const _DependancyLayer& get_data_dependancy() const;
  const _DependancyLayer& get_ctrl_dependancy() const;
  const _DependancyLayer& get_enabled_dependancy() const;

  // Warning: It is user's responsibility to make sure different layer's
  // dependancy are consistent.
  _DependancyLayer& get_data_dependancy_mut();
  _DependancyLayer& get_ctrl_dependancy_mut();
  _DependancyLayer& get_enabled_dependancy_mut();

 private:
  bool enable_data_deps;
  bool enable_ctrl_deps;
  _DependancyLayer data_dependancy;
  _DependancyLayer ctrl_dependancy;
  _DependancyLayer enabled_dependancy;

  // §23 mem opt: tracks whether any ctrl_dep edge was seen during add_node.
  // Llama/Qwen/GPT STG-generated workloads never emit ctrl_deps (only
  // data_deps).  In that case, after resolve, enabled_dependancy == data_dependancy
  // and ctrl_dependancy is pure bookkeeping (all nodes dep-free from the
  // start with no transitions).  We drop both redundant layers to free ~2/3
  // of the DependancyResolver memory.  When any ctrl edge is present we
  // fall back to the three-layer behaviour unchanged.
  bool _alias_enabled_to_data = false;
  uint64_t _total_ctrl_edges = 0;
};

} // namespace FeederV3
} // namespace Chakra

#endif
