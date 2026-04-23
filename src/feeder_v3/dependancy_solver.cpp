#include "dependancy_solver.h"
#include <algorithm>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace Chakra::FeederV3;

namespace {
// Helper: remove `val` from `vec` using swap-and-pop (O(1) after find).
// Returns true iff val was present and removed.  Order of remaining elements
// is not preserved, which is fine — the adjacency lists are unordered sets
// semantically.
inline bool erase_swap(NodeIdList& vec, NodeId val) {
  auto it = std::find(vec.begin(), vec.end(), val);
  if (it == vec.end()) return false;
  if (it + 1 != vec.end()) *it = vec.back();
  vec.pop_back();
  return true;
}
}  // namespace

void _DependancyLayer::add_node(
    const NodeId& node,
    const std::unordered_set<NodeId>& parents) {
  std::unique_lock<std::shared_mutex> lock(this->mutex);
  this->dirty = true;
  this->_helper_allocate_bucket(node);
  auto& node_parents = this->child_map_parent[node];
  // Reserve to avoid repeated reallocations when many edges attach to
  // the same node.
  if (node_parents.capacity() < node_parents.size() + parents.size())
    node_parents.reserve(node_parents.size() + parents.size());
  for (auto& parent : parents) {
    this->_helper_allocate_bucket(parent);
    _push_unique(this->child_map_parent[node], parent);
    _push_unique(this->parent_map_child[parent], node);
  }
}

void _DependancyLayer::add_node_children(
    const NodeId& node,
    const std::unordered_set<NodeId>& children) {
  std::unique_lock<std::shared_mutex> lock(this->mutex);
  this->dirty = true;
  this->_helper_allocate_bucket(node);
  auto& node_children = this->parent_map_child[node];
  if (node_children.capacity() < node_children.size() + children.size())
    node_children.reserve(node_children.size() + children.size());
  for (auto& child : children) {
    this->_helper_allocate_bucket(child);
    _push_unique(this->child_map_parent[child], node);
    _push_unique(this->parent_map_child[node], child);
  }
}

void _DependancyLayer::take_node(const NodeId& node) {
  std::unique_lock<std::shared_mutex> lock(this->mutex);
  if (this->dirty) {
    throw std::runtime_error(
        "dependancy layer is dirty, resolve_dependancy_free_nodes should be called first");
  }
  if (this->dependancy_free_nodes.find(node) ==
      this->dependancy_free_nodes.end()) {
    throw std::runtime_error(
        "Node " + std::to_string(node) +
        " is not dependancy free or already taken/released");
  }
  if (this->ongoing_nodes.find(node) != this->ongoing_nodes.end()) {
    throw std::runtime_error("Node is already taken");
  }
  this->ongoing_nodes.insert(node);
  this->dependancy_free_nodes.erase(node);
}

void _DependancyLayer::finish_node(const NodeId& node) {
  std::unique_lock<std::shared_mutex> lock(this->mutex);
  if (this->dirty) {
    throw std::runtime_error(
        "dependancy layer is dirty, resolve_dependancy_free_nodes should be called first");
  }
  if (this->ongoing_nodes.find(node) == this->ongoing_nodes.end()) {
    throw std::runtime_error("Node is not taken");
  }
  this->ongoing_nodes.erase(node);
  // Take a copy of the children list because we're about to erase `node`
  // from parent_map_child.  The inner loop mutates child_map_parent[child].
  auto it_pmc = this->parent_map_child.find(node);
  if (it_pmc != this->parent_map_child.end()) {
    for (const NodeId& child : it_pmc->second) {
      auto it_cmp = this->child_map_parent.find(child);
      if (it_cmp == this->child_map_parent.end()) {
        throw std::runtime_error(
            "Parent map child is not consistent with child map parent");
      }
      if (!erase_swap(it_cmp->second, node)) {
        throw std::runtime_error(
            "Parent map child is not consistent with child map parent");
      }
      if (it_cmp->second.empty()) {
        this->dependancy_free_nodes.insert(child);
      }
    }
  }
  this->child_map_parent.erase(node);
  this->parent_map_child.erase(node);
}

void _DependancyLayer::push_back_node(const NodeId& node) {
  std::unique_lock<std::shared_mutex> lock(this->mutex);
  if (this->dirty) {
    throw std::runtime_error(
        "dependancy layer is dirty, resolve_dependancy_free_nodes should be called first");
  }
  if (this->ongoing_nodes.find(node) == this->ongoing_nodes.end()) {
    throw std::runtime_error("Node is not taken");
  }
  this->ongoing_nodes.erase(node);
  this->dependancy_free_nodes.insert(node);
}

void _DependancyLayer::resolve_dependancy_free_nodes() {
  std::unique_lock<std::shared_mutex> lock(this->mutex);
  if ((!this->dependancy_free_nodes.empty()) || (!this->ongoing_nodes.empty()))
    throw std::runtime_error(
        "resolve_dependancy_free_nodes after initialization is not supported yet!");
  // Release excess adjacency-list capacity now that graph construction is
  // complete; shrink_to_fit trims each node's vector buffer down to its
  // actual size (typically 1-5 NodeIds each).  On 16-rank llama this alone
  // removes ~15% of resident memory after load.
  for (auto& it : this->child_map_parent) {
    if (it.second.empty())
      this->dependancy_free_nodes.insert(it.first);
    it.second.shrink_to_fit();
  }
  for (auto& it : this->parent_map_child) {
    it.second.shrink_to_fit();
  }
  if (this->dependancy_free_nodes.empty())
    throw std::runtime_error(
        "No dependancy free nodes found, there might be deadlocks");
  this->dirty = false;
}

const std::unordered_set<NodeId>& _DependancyLayer::get_dependancy_free_nodes()
    const {
  return this->dependancy_free_nodes;
}

const NodeIdList& _DependancyLayer::get_children(
    NodeId node) const {
  return this->parent_map_child.at(node);
}

const NodeIdList& _DependancyLayer::get_parents(
    NodeId node) const {
  return this->child_map_parent.at(node);
}

const std::unordered_set<NodeId>& _DependancyLayer::get_ongoing_nodes() const {
  return this->ongoing_nodes;
}

void _DependancyLayer::_helper_allocate_bucket(NodeId node_id) {
  // Calling operator[] inserts a default-constructed (empty) vector if the
  // key isn't present — cheaper than the previous "find first" pattern
  // because unordered_map::find + operator[] was doing two hash lookups.
  this->child_map_parent[node_id];
  this->parent_map_child[node_id];
}

void _DependancyLayer::reset() {
  // Swap-with-empty to truly release bucket memory (unordered_map::clear()
  // keeps the bucket array alive).
  std::unique_lock<std::shared_mutex> lock(this->mutex);
  std::unordered_map<NodeId, NodeIdList>().swap(this->child_map_parent);
  std::unordered_map<NodeId, NodeIdList>().swap(this->parent_map_child);
  std::unordered_set<NodeId>().swap(this->dependancy_free_nodes);
  std::unordered_set<NodeId>().swap(this->ongoing_nodes);
  this->dirty = true;
}

void DependancyResolver::add_node(const ChakraNode& node) {
  NodeId node_id = node.id();
  std::unordered_set<NodeId> parents, enabled_parents;
  for (auto& parent : node.data_deps()) {
    if (this->enable_data_deps)
      enabled_parents.insert(parent);
    parents.insert(parent);
  }
  this->data_dependancy.add_node(node_id, parents);
  parents.clear();

  for (auto& parent : node.ctrl_deps()) {
    if (this->enable_ctrl_deps)
      enabled_parents.insert(parent);
    parents.insert(parent);
  }
  this->_total_ctrl_edges += parents.size();
  this->ctrl_dependancy.add_node(node_id, parents);
  parents.clear();

  this->enabled_dependancy.add_node(node_id, enabled_parents);
}

void DependancyResolver::take_node(const NodeId& node) {
  this->data_dependancy.take_node(node);
  if (this->_alias_enabled_to_data) return;
  this->ctrl_dependancy.take_node(node);
  this->enabled_dependancy.take_node(node);
}

void DependancyResolver::push_back_node(const NodeId& node) {
  this->data_dependancy.push_back_node(node);
  if (this->_alias_enabled_to_data) return;
  this->ctrl_dependancy.push_back_node(node);
  this->enabled_dependancy.push_back_node(node);
}

void DependancyResolver::finish_node(const NodeId& node) {
  this->data_dependancy.finish_node(node);
  if (this->_alias_enabled_to_data) return;
  this->ctrl_dependancy.finish_node(node);
  this->enabled_dependancy.finish_node(node);
}

void DependancyResolver::resolve_dependancy_free_nodes() {
  this->data_dependancy.resolve_dependancy_free_nodes();
  // §23 mem opt: when no ctrl_dep edges exist and both flags are enabled
  // (the default for STG-generated llama/gpt/qwen workloads), the ctrl
  // and enabled layers are pure-bookkeeping duplicates of data.  Drop
  // them: calling each layer's resolve first so their state is self-
  // consistent (no throws), then swap their contents into scratch objects
  // that go out of scope and release their memory.  From here on,
  // take/finish/push_back skip the two dropped layers; get_* routes all
  // queries to data_dependancy via the alias flag.
  this->ctrl_dependancy.resolve_dependancy_free_nodes();
  this->enabled_dependancy.resolve_dependancy_free_nodes();
  if (this->_total_ctrl_edges == 0 &&
      this->enable_data_deps && this->enable_ctrl_deps) {
    // Redundant layers: enabled_parents == data_parents when ctrl_deps
    // is empty.  Release their memory entirely and route all future
    // take/finish/push_back/get_* queries to data_dependancy.
    this->ctrl_dependancy.reset();
    this->enabled_dependancy.reset();
    this->_alias_enabled_to_data = true;
  }
}

const std::unordered_set<NodeId>& DependancyResolver::
    get_dependancy_free_nodes() const {
  if (this->_alias_enabled_to_data)
    return this->data_dependancy.get_dependancy_free_nodes();
  return this->enabled_dependancy.get_dependancy_free_nodes();
}

const std::unordered_set<NodeId>& DependancyResolver::get_ongoing_nodes()
    const {
  if (this->_alias_enabled_to_data)
    return this->data_dependancy.get_ongoing_nodes();
  return this->enabled_dependancy.get_ongoing_nodes();
}

const _DependancyLayer& DependancyResolver::get_data_dependancy() const {
  return this->data_dependancy;
}

const _DependancyLayer& DependancyResolver::get_ctrl_dependancy() const {
  // When aliased to data, the ctrl layer was reset to release memory; but
  // the return reference still lives on `this` so the empty layer is safe
  // to return.  Callers should not assume ctrl content when aliased.
  return this->ctrl_dependancy;
}

const _DependancyLayer& DependancyResolver::get_enabled_dependancy() const {
  if (this->_alias_enabled_to_data)
    return this->data_dependancy;
  return this->enabled_dependancy;
}

_DependancyLayer& DependancyResolver::get_data_dependancy_mut() {
  return this->data_dependancy;
}

_DependancyLayer& DependancyResolver::get_ctrl_dependancy_mut() {
  return this->ctrl_dependancy;
}

_DependancyLayer& DependancyResolver::get_enabled_dependancy_mut() {
  return this->enabled_dependancy;
}
