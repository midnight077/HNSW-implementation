# FAISS `IndexHNSW.cpp` — Deep Dive (Beginner Friendly)

This document walks through the two most important operations in FAISS's HNSW implementation:

1. **Building the Index** — adding vectors so the graph is constructed
2. **Searching a Query** — finding the nearest neighbours for a new vector

---

## 🏗️ Part 1: Building the Index (Index Construction)

### The Big Picture

When you call `index.add(vectors)` in FAISS, you are not just storing vectors in a list. You are **weaving them into a multi-layer graph** — the HNSW graph — where each vector becomes a node connected to its nearest neighbours at various levels. This is what makes search lightning fast later.

```
User Code
   │
   ▼
IndexHNSW::add()          ← Entry point (line 378)
   │
   ├─► storage->add()     ← Save the raw vectors to flat storage
   │
   └─► hnsw_add_vertices() ← Build the graph connections (line 63)
           │
           ├─ Step 1: Assign each vector a random level
           ├─ Step 2: Sort vectors highest-level-first (bucket sort)
           └─ Step 3: For each vector, call hnsw.add_with_locks()
```

---

### Step-by-Step: `IndexHNSW::add()` (lines 378–394)

```cpp
void IndexHNSW::add(idx_t n, const float* x) {
    size_t n0 = ntotal;          // how many vectors exist already
    storage->add(n, x);          // save raw vectors
    ntotal = storage->ntotal;    // update total count

    hnsw_add_vertices(*this, n0, n, x, verbose, ...);
}
```

**Plain English:**
- `n` = number of new vectors you're adding
- `x` = pointer to the raw float data (a flat array of all vectors)
- `n0` = the "old" count — this is used to give new vectors unique IDs (`n0`, `n0+1`, …, `n0+n-1`)
- First, the raw vector data is stored in `storage` (which is just a flat array / compressed store)
- Then the real work begins: building graph edges via `hnsw_add_vertices`

---

### Step-by-Step: `hnsw_add_vertices()` (lines 63–213)

This is the core function that builds the HNSW graph. It has three major phases.

---

#### Phase 1: Assign Levels (line 86)

```cpp
int max_level = hnsw.prepare_level_tab(n, preset_levels);
```

**What happens here?**

Each vector in HNSW lives on one or more "floors" of a multi-story building (the layers). The **level** of a vector is chosen **randomly**, following an exponential distribution — most vectors get level 0 (the ground floor), fewer get level 1, even fewer get level 2, and so on.

> **Analogy:** Imagine a hotel. Most guests stay on the ground floor. A few are VIPs on floor 1. Even fewer are on floor 2. The higher the floor, the fewer nodes, and the bigger the "express highway" jumps during search.

This is pre-computed and stored in `hnsw.levels[]`.

---

#### Phase 2: Bucket Sort (lines 99–123)

```cpp
// Build histogram of how many vectors are at each level
for (size_t i = 0; i < n; i++) {
    int pt_level = hnsw.levels[pt_id] - 1;
    hist[pt_level]++;
}

// Sort vectors so highest-level vectors come first
for (size_t i = 0; i < n; i++) {
    order[offsets[pt_level]++] = pt_id;
}
```

**Why sort?**

HNSW requires that when you insert a vector at level `L`, the graph at levels `L+1`, `L+2`, … already exists — because you need to navigate those higher levels to find the best entry point into the current level.

So vectors at **higher levels must be inserted first**. This bucket sort groups them into per-level buckets and processes them highest → lowest.

> **Analogy:** You're building a city's road network. You must build the highway (top level) before the local streets (bottom level), so that when you build local streets you can already use the highway to navigate efficiently.

---

#### Phase 3: Parallel Insertion (lines 128–205)

```cpp
for (int pt_level = hist.size()-1; pt_level >= 0; pt_level--) {
    // For each batch of vectors at this level...

#pragma omp parallel          // ← uses multiple CPU threads
    {
        DistanceComputer dis = storage_distance_computer(storage);

        for (each vector in this level's batch) {
            dis->set_query(x + (pt_id - n0) * d);  // point to this vector's data
            
            hnsw.add_with_locks(dis, pt_level, pt_id, locks, vt, ...);
        }
    }
}
```

**What does `hnsw.add_with_locks()` do?** (this lives in `HNSW.cpp`, called from here)

For each new vector being inserted, it:
1. **Finds the best entry point** — starts from the current graph entry point at the highest level
2. **Greedy descends** through the layers until it reaches the new vector's max level
3. **At each layer (from max level down to 0)**: runs a beam search with `efConstruction` candidates to find the `M` best neighbours
4. **Creates bidirectional edges** between the new vector and its selected neighbours
5. **Shrinks neighbour lists** if they exceed `M` (or `2M` at level 0) using the heuristic algorithm

**Key parameters:**
| Parameter | Meaning |
|-----------|---------|
| `M` | Max connections per node per layer (default: 16 or 32) |
| `efConstruction` | Size of the dynamic candidate list during insertion — higher = better quality, slower build |
| `pt_level` | The level being processed in this iteration |

**Thread Safety (`locks`):**

Because OMP is running this in parallel (multiple threads inserting different vectors simultaneously), **locks** are used per-node. When thread A is adding edges to node 5, thread B must wait before it can also touch node 5. This prevents race conditions in the graph.

---

#### Random Permutation (lines 143–147)

```cpp
for (size_t j = i0; j < i1; j++) {
    std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);
}
```

Within each level-batch, the vectors are **randomly shuffled** before insertion. This prevents any ordering bias from the original dataset from affecting graph quality (e.g., if vectors were already sorted by some feature).

---

### Summary: Index Construction Flow

```
IndexHNSW::add(n, x)
│
├── storage->add(n, x)
│     └── Stores raw float vectors in flat storage (for distance computation later)
│
└── hnsw_add_vertices(...)
      │
      ├── [Phase 1] hnsw.prepare_level_tab()
      │     └── For each new vector: randomly draw level from exponential dist.
      │           Most → level 0, few → level 1, very few → level 2, ...
      │
      ├── [Phase 2] Bucket sort by level
      │     └── order[] = sorted list: highest level vectors come first
      │
      └── [Phase 3] Process highest level → lowest level
            └── For each batch at a given level (in parallel):
                  ├── set_query(vector data)
                  └── hnsw.add_with_locks()
                        ├── Navigate from entry point down to this vector's level
                        ├── Beam search for M best neighbours at each layer
                        ├── Create edges (bidirectional)
                        └── Shrink neighbour lists if over capacity
```

---

---

## 🔍 Part 2: Searching a Query

### The Big Picture

When you call `index.search(query, k)`, FAISS navigates the HNSW graph from the **top layer down**, using a greedy best-first strategy to efficiently zoom in on the `k` nearest neighbours.

```
User Code
   │
   ▼
IndexHNSW::search()        ← Entry point (line 328)
   │
   └─► hnsw_search()       ← Core search loop (line 252)
           │
           └─► hnsw.search()  ← Layer-by-layer greedy descent (in HNSW.cpp)
```

---

### Step-by-Step: `IndexHNSW::search()` (lines 328–346)

```cpp
void IndexHNSW::search(idx_t n, const float* x, idx_t k,
                        float* distances, idx_t* labels,
                        const SearchParameters* params) const {

    if (is_similarity_metric(this->metric_type)) {
        using RH = HeapBlockResultHandler<HNSW::C_similarity>;
        RH bres(n, distances, labels, k);
        hnsw_search(this, n, x, bres, params);
    } else {
        using RH = HeapBlockResultHandler<HNSW::C_distance>;
        RH bres(n, distances, labels, k);
        hnsw_search(this, n, x, bres, params);
    }
}
```

**Plain English:**
- `n` = number of query vectors to search for
- `x` = the query vectors (flat float array)
- `k` = how many nearest neighbours to return
- `distances` = output array: will be filled with the k distances
- `labels` = output array: will be filled with the k vector IDs

The only difference in the two branches is **how results are ranked**:
- `C_distance` → **smaller distance = better** (used for L2 / Euclidean)
- `C_similarity` → **larger score = better** (used for Inner Product / cosine)

Both use a **max-heap** internally to track the top-k results so far. Think of it as a "priority waiting room" that always keeps the k best candidates seen.

---

### Step-by-Step: `hnsw_search()` (lines 252–324)

```cpp
template <class BlockResultHandler>
void hnsw_search(const IndexHNSW* index, idx_t n, const float* x,
                 BlockResultHandler& bres, const SearchParameters* params) {

    int efSearch = hnsw.efSearch;   // how wide the beam is during search

    // Process queries in periodic batches (for interrupt checking)
    for (idx_t i0 = 0; i0 < n; i0 += check_period) {

#pragma omp parallel        // ← multiple threads, one per query
        {
            VisitedTable vt = ...;     // tracks which nodes we've visited
            DistanceComputer dis = ...; // computes distances to query

            for (idx_t i = i0; i < i1; i++) {  // each query
                res->begin(i);
                dis->set_query(x + i * index->d);  // point to query vector i

                HNSWStats stats = hnsw.search(*dis, index, *res, *vt, params);
                
                res->end();
                vt->advance();   // reset visited table for next query
            }
        }
    }
}
```

**Key points:**
- Each query is handled by one thread (OMP parallel for)
- `VisitedTable` keeps a fast bitmask of which graph nodes have been visited — avoids revisiting nodes
- `DistanceComputer` is a function object that computes the distance from the query to any stored vector
- `hnsw.search()` is where the actual layer-by-layer greedy walk happens

---

### What `hnsw.search()` Does Internally (conceptual)

> This function is in `HNSW.cpp`, but here's what it does for the search:

**Step 1 — Enter at the Top Layer**

Start at the single "entry point" node of the graph (this is the node with the highest level). Compute the distance from this node to the query.

**Step 2 — Greedy Descend (Upper Layers)**

For all layers from `max_level` down to `1`:
- Look at all neighbours of the current node at this layer
- Move to whichever neighbour is closest to the query
- Repeat (greedy best-first) until no neighbour is closer → you've found the local minimum
- Use that node as the entry point for the next (lower) layer

> **Analogy:** You're looking for a coffee shop on a map. You first zoom out (high layer) and quickly jump city-to-city until you're in the right city. Then you zoom in (lower layer) and jump neighbourhood-to-neighbourhood. Finally at street level (layer 0) you do a careful local search.

**Step 3 — Beam Search at Layer 0 (The Final Layer)**

At the base layer, instead of greedy 1-best, use a **beam search** with `efSearch` candidates:
- Maintain a candidate priority queue (min-heap by distance)
- Maintain a result heap (max-heap of top-k found so far)
- Pop the closest candidate, explore its neighbours
- Add unvisited neighbours to the candidate queue
- Stop when the closest candidate is farther than the worst result in the top-k heap

**Key parameter:**
| Parameter | Meaning |
|-----------|---------|
| `efSearch` | Beam width at layer 0. Higher = more accurate but slower. Must be ≥ k. |

---

### The `VisitedTable` (line 153, 288)

```cpp
std::unique_ptr<VisitedTable> vt = VisitedTable::create(ntotal, hnsw.use_visited_hashset);
// ... after each query:
vt->advance();  // resets the table cheaply without re-allocating
```

This is a **O(1) visited check** mechanism. Instead of clearing a boolean array (O(n)) between queries, it uses a "generation counter" trick: a node is "visited" if its stored generation equals the current generation. `advance()` just increments the counter.

---

### Result Collection: `HeapBlockResultHandler`

```cpp
RH bres(n, distances, labels, k);
```

This is the output buffer. It wraps the `distances[]` and `labels[]` arrays you passed in. It maintains a **max-heap of size k** per query:
- If we find a new vector closer than the worst in the heap → replace the worst
- At the end, `end()` is called which sorts the heap (best first)

---

### Summary: Search Flow

```
IndexHNSW::search(n, x, k, distances, labels)
│
├── Create HeapBlockResultHandler (output buffer, k-max-heap per query)
│
└── hnsw_search(...)
      │
      └── For each query i (in parallel):
            │
            ├── set_query(x[i])      — point distance computer to query vector
            │
            └── hnsw.search(...)     — the graph walk
                  │
                  ├── [Upper layers 1..max_level]
                  │     └── Greedy 1-best descent
                  │           Start at entry point, move to closest neighbour
                  │           Repeat until no improvement → descend to next layer
                  │
                  └── [Layer 0 — Final beam search]
                        ├── Candidate min-heap (size = efSearch)
                        ├── Result max-heap (size = k)
                        ├── Pop closest candidate
                        │     ├── If its distance > worst in result heap → STOP
                        │     └── Else explore its neighbours:
                        │           ├── Skip if already visited (VisitedTable)
                        │           ├── Compute distance to query
                        │           ├── Add to candidates if promising
                        │           └── Update result heap if it's in top-k
                        └── Return top-k (ids + distances)
```

---

## 🔑 Key Parameters Quick Reference

| Parameter | Used In | Meaning |
|-----------|---------|---------|
| `M` | Build | Max edges per node per layer. Higher → better graph quality, more memory |
| `efConstruction` | Build | Beam width during insertion. Higher → better quality graph, slower build |
| `efSearch` | Search | Beam width at layer 0. Higher → more accurate, slower search |
| `max_level` | Both | Highest layer in the current graph |
| `ntotal` | Both | Total number of vectors currently in the index |
| `d` | Both | Dimensionality of the vectors |

## 📐 The Two Separate Concerns

| Concern | Where Stored | Purpose |
|---------|-------------|---------|
| **Raw Vectors** | `storage` (IndexFlat etc.) | Computing distances during search and build |
| **Graph Structure** | `hnsw.neighbors[]`, `hnsw.levels[]` | Navigation during search and build |

These are intentionally kept separate so you can swap the storage backend (e.g., use scalar quantized vectors for distance computation) without changing the graph logic.
