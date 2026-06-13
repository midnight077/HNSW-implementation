// ============================================================================
// HNSW (Hierarchical Navigable Small World) — Simple C++ Implementation
// ============================================================================
// This is a teaching implementation. It prioritizes clarity over performance.
// It implements the three core operations:
//   1. Graph construction (inserting vectors one by one)
//   2. Inserting a new vector into an existing graph
//   3. Approximate Nearest Neighbor (ANN) search
//
// Compile: g++ -std=c++17 -o hnsw hnsw.cpp -lm
// Run:     ./hnsw
// ============================================================================

#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <string>
#include <iomanip>
#include <functional>

// ============================================================================
// Section 1: Data Structures
// ============================================================================

struct Node {
    int id;
    std::string label;                                    // human-readable name
    std::vector<float> vec;                               // the embedding vector
    int max_layer;                                        // highest layer this node lives on
    std::vector<std::vector<int>> neighbors;              // neighbors[layer] = list of neighbor IDs
};

// A candidate during search: (distance, node_id)
// We use this with priority queues.
using Candidate = std::pair<float, int>;

// ============================================================================
// Section 2: The HNSW Index
// ============================================================================

class HNSWIndex {
public:
    // --- Parameters ---
    int M;                  // max neighbors per node per layer
    int M_max0;             // max neighbors at layer 0 (typically 2*M)
    int ef_construction;    // candidate list size during insertion
    float mL;               // level multiplier: 1/ln(M)

    // --- Graph state ---
    std::vector<Node> nodes;
    int entry_point;        // ID of the current entry point node
    int max_level;          // highest layer in the graph

    // --- RNG ---
    std::mt19937 rng;

    // Constructor
    HNSWIndex(int M = 3, int ef_construction = 8, unsigned seed = 42)
        : M(M), ef_construction(ef_construction), entry_point(-1), max_level(-1), rng(seed)
    {
        M_max0 = 2 * M;             // layer 0 allows more connections
        mL = 1.0f / std::log(M);    // normalization factor for level generation
    }

    // ========================================================================
    // Core Operation 1: Euclidean Distance
    // ========================================================================
    // dist(a, b) = sqrt( sum_i (a_i - b_i)^2 )
    float distance(const std::vector<float>& a, const std::vector<float>& b) {
        float sum = 0.0f;
        for (size_t i = 0; i < a.size(); i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    // ========================================================================
    // Core Operation 2: Random Layer Assignment
    // ========================================================================
    // Formula: l = floor( -ln(uniform(0,1)) * mL )
    // This creates an exponential distribution — most nodes get layer 0,
    // fewer get layer 1, very few get layer 2, etc.
    int random_level() {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng);
        int level = static_cast<int>(-std::log(r) * mL);
        return level;
    }

    // ========================================================================
    // Core Operation 3: Greedy Search Within a Single Layer
    // ========================================================================
    // Starting from 'entry_points', greedily explore the graph at 'layer'
    // to find the 'ef' closest nodes to 'query'.
    //
    // Returns: a max-heap of (distance, node_id) — the ef closest candidates.
    //
    // Algorithm:
    //   1. Initialize candidate list (min-heap) and result list (max-heap)
    //   2. Pop the closest unvisited candidate
    //   3. If it's farther than the farthest result, stop (we can't improve)
    //   4. Otherwise, check all its neighbors — add closer ones to candidates
    //   5. Keep result list trimmed to size 'ef'
    std::priority_queue<Candidate> search_layer(
        const std::vector<float>& query,
        std::vector<int> entry_points,
        int ef,
        int layer
    ) {
        std::unordered_set<int> visited;

        // Min-heap for candidates (closest first)
        auto cmp_min = [](const Candidate& a, const Candidate& b) {
            return a.first > b.first;
        };
        std::priority_queue<Candidate, std::vector<Candidate>, decltype(cmp_min)>
            candidates(cmp_min);

        // Max-heap for results (farthest first, so we can trim)
        std::priority_queue<Candidate> results;

        // Seed with entry points
        for (int ep : entry_points) {
            float d = distance(query, nodes[ep].vec);
            candidates.push({d, ep});
            results.push({d, ep});
            visited.insert(ep);
        }

        while (!candidates.empty()) {
            auto [cand_dist, cand_id] = candidates.top();
            candidates.pop();

            // If closest candidate is farther than our worst result, stop
            float worst_dist = results.top().first;
            if (cand_dist > worst_dist) break;

            // Explore neighbors of this candidate at the given layer
            if (layer < (int)nodes[cand_id].neighbors.size()) {
                for (int neighbor_id : nodes[cand_id].neighbors[layer]) {
                    if (visited.count(neighbor_id)) continue;
                    visited.insert(neighbor_id);

                    float d = distance(query, nodes[neighbor_id].vec);
                    worst_dist = results.top().first;

                    // Add if closer than our worst result, or if we have room
                    if (d < worst_dist || (int)results.size() < ef) {
                        candidates.push({d, neighbor_id});
                        results.push({d, neighbor_id});
                        if ((int)results.size() > ef) {
                            results.pop();  // remove the farthest
                        }
                    }
                }
            }
        }
        return results;  // max-heap of the ef closest nodes
    }

    // ========================================================================
    // Core Operation 4: INSERT a vector into the HNSW graph
    // ========================================================================
    // This is the heart of HNSW construction. Every call to insert() builds
    // the graph incrementally — construction is just repeated insertion.
    //
    // Steps:
    //   1. Assign a random layer to the new node
    //   2. Greedy descent from top layer to (assigned_layer + 1) with ef=1
    //   3. At assigned_layer down to 0: search with ef=ef_construction,
    //      connect to M nearest neighbors
    //   4. If new node's layer > current max, update entry point
    int insert(const std::string& label, const std::vector<float>& vec) {
        // Create the new node
        int new_id = nodes.size();
        int new_level = random_level();

        Node new_node;
        new_node.id = new_id;
        new_node.label = label;
        new_node.vec = vec;
        new_node.max_layer = new_level;
        new_node.neighbors.resize(new_level + 1);  // one neighbor list per layer
        nodes.push_back(new_node);

        // --- Special case: first node ever ---
        if (entry_point == -1) {
            entry_point = new_id;
            max_level = new_level;
            std::cout << "  Inserted [" << label << "] as first node (layer " << new_level << ")\n";
            return new_id;
        }

        int ep = entry_point;

        // --- Phase 1: Greedy descent from top to (new_level + 1) ---
        // Use ef=1 (single nearest neighbor) — we're just navigating, not connecting
        for (int lc = max_level; lc > new_level; lc--) {
            auto results = search_layer(vec, {ep}, 1, lc);
            ep = results.top().second;  // closest node at this layer
        }

        // --- Phase 2: Search and connect at layers min(max_level, new_level) ... 0 ---
        for (int lc = std::min(max_level, new_level); lc >= 0; lc--) {
            auto results = search_layer(vec, {ep}, ef_construction, lc);

            // Extract candidates sorted by distance (closest first)
            std::vector<Candidate> candidates;
            while (!results.empty()) {
                candidates.push_back(results.top());
                results.pop();
            }
            std::sort(candidates.begin(), candidates.end());

            // Select the M closest neighbors
            int max_conn = (lc == 0) ? M_max0 : M;
            int num_neighbors = std::min((int)candidates.size(), max_conn);

            for (int i = 0; i < num_neighbors; i++) {
                int neighbor_id = candidates[i].second;

                // Bidirectional connection: new_node <-> neighbor
                nodes[new_id].neighbors[lc].push_back(neighbor_id);
                // Ensure neighbor has enough layers in its neighbor list
                while ((int)nodes[neighbor_id].neighbors.size() <= lc) {
                    nodes[neighbor_id].neighbors.push_back({});
                }
                nodes[neighbor_id].neighbors[lc].push_back(new_id);

                // Prune neighbor if it exceeds max connections
                int neighbor_max = (lc == 0) ? M_max0 : M;
                if ((int)nodes[neighbor_id].neighbors[lc].size() > neighbor_max) {
                    prune_connections(neighbor_id, lc, neighbor_max);
                }
            }

            // Use the closest candidate as entry point for the next layer down
            if (!candidates.empty()) {
                ep = candidates[0].second;
            }
        }

        // --- Phase 3: Update entry point if new node is the highest ---
        if (new_level > max_level) {
            entry_point = new_id;
            max_level = new_level;
        }

        std::cout << "  Inserted [" << label << "] at layer " << new_level << "\n";
        return new_id;
    }

    // ========================================================================
    // Core Operation 5: ANN SEARCH — find K approximate nearest neighbors
    // ========================================================================
    // Steps:
    //   1. Start at the entry point on the top layer
    //   2. Greedy descent (ef=1) from top layer to layer 1
    //   3. At layer 0: full search with ef=ef_search candidates
    //   4. Return the top K results
    std::vector<Candidate> search(const std::vector<float>& query, int K, int ef_search) {
        if (entry_point == -1) return {};

        int ep = entry_point;

        // Phase 1: Greedy descent from top to layer 1
        for (int lc = max_level; lc > 0; lc--) {
            auto results = search_layer(query, {ep}, 1, lc);
            ep = results.top().second;
        }

        // Phase 2: Search layer 0 with full ef budget
        auto results = search_layer(query, {ep}, std::max(ef_search, K), 0);

        // Extract and sort results
        std::vector<Candidate> top_k;
        while (!results.empty()) {
            top_k.push_back(results.top());
            results.pop();
        }
        std::sort(top_k.begin(), top_k.end());  // sort by distance ascending

        // Return only top K
        if ((int)top_k.size() > K) top_k.resize(K);
        return top_k;
    }

    // ========================================================================
    // Helper: Prune a node's connections to keep only the closest ones
    // ========================================================================
    void prune_connections(int node_id, int layer, int max_conn) {
        auto& neighbors = nodes[node_id].neighbors[layer];
        std::vector<Candidate> scored;
        for (int nid : neighbors) {
            float d = distance(nodes[node_id].vec, nodes[nid].vec);
            scored.push_back({d, nid});
        }
        std::sort(scored.begin(), scored.end());  // closest first
        neighbors.clear();
        for (int i = 0; i < std::min((int)scored.size(), max_conn); i++) {
            neighbors.push_back(scored[i].second);
        }
    }

    // ========================================================================
    // Visualization: Print the graph structure
    // ========================================================================
    void print_graph() {
        std::cout << "\n========================================\n";
        std::cout << "  HNSW Graph Structure\n";
        std::cout << "  Entry Point: " << nodes[entry_point].label
                  << " | Max Layer: " << max_level << "\n";
        std::cout << "========================================\n";

        for (int layer = max_level; layer >= 0; layer--) {
            std::cout << "\n--- Layer " << layer << " ---\n";
            for (auto& node : nodes) {
                if (node.max_layer >= layer) {
                    std::cout << "  " << std::setw(8) << node.label
                              << " [" << node.vec[0] << "," << node.vec[1] << "] -> { ";
                    if (layer < (int)node.neighbors.size()) {
                        for (int nid : node.neighbors[layer]) {
                            std::cout << nodes[nid].label << " ";
                        }
                    }
                    std::cout << "}\n";
                }
            }
        }
        std::cout << "\n";
    }
};

// ============================================================================
// Main: Demonstrate all three operations with the Food example
// ============================================================================
int main() {
    std::cout << "==========================================================\n";
    std::cout << "  HNSW Demo: Food Embeddings [sweetness, spiciness]\n";
    std::cout << "  Parameters: M=3, efConstruction=8, mL=1/ln(3)≈0.91\n";
    std::cout << "==========================================================\n\n";

    HNSWIndex index(/*M=*/3, /*ef_construction=*/8, /*seed=*/42);

    // ---- PART 1: Build the graph by inserting 8 food items ----
    std::cout << "--- Part 1: Graph Construction ---\n";
    index.insert("Apple",  {9.0f,  1.0f});
    index.insert("Banana", {8.0f,  0.0f});
    index.insert("Mango",  {10.0f, 2.0f});
    index.insert("Pepper", {2.0f,  10.0f});
    index.insert("Wasabi", {1.0f,  9.0f});
    index.insert("Ginger", {3.0f,  7.0f});
    index.insert("Honey",  {10.0f, 0.0f});
    index.insert("Lemon",  {5.0f,  4.0f});

    index.print_graph();

    // ---- PART 2: Insert a NEW vector into the existing graph ----
    std::cout << "--- Part 2: Inserting a New Vector ---\n";
    index.insert("Jalapeno", {3.0f, 9.0f});
    index.print_graph();

    // ---- PART 3: ANN Search ----
    std::cout << "--- Part 3: ANN Search ---\n";
    std::vector<float> query = {4.0f, 8.0f};  // "Sriracha"
    int K = 3;
    int ef_search = 10;

    std::cout << "Query: Sriracha [4, 8]\n";
    std::cout << "Finding " << K << " approximate nearest neighbors (ef=" << ef_search << "):\n\n";

    auto results = index.search(query, K, ef_search);

    std::cout << std::fixed << std::setprecision(2);
    for (int i = 0; i < (int)results.size(); i++) {
        auto [dist, id] = results[i];
        std::cout << "  #" << (i+1) << "  " << std::setw(8) << index.nodes[id].label
                  << " [" << index.nodes[id].vec[0] << "," << index.nodes[id].vec[1] << "]"
                  << "  distance = " << dist << "\n";
    }

    // ---- Verify against brute force ----
    std::cout << "\n--- Brute Force Verification ---\n";
    std::vector<Candidate> brute;
    for (auto& node : index.nodes) {
        float d = index.distance(query, node.vec);
        brute.push_back({d, node.id});
    }
    std::sort(brute.begin(), brute.end());

    std::cout << "All distances from Sriracha [4,8]:\n";
    for (auto& [d, id] : brute) {
        std::cout << "  " << std::setw(8) << index.nodes[id].label
                  << "  distance = " << d << "\n";
    }

    return 0;
}
