# HNSW: Hierarchical Navigable Small World — A Complete Walkthrough

## Table of Contents
1. [The Problem: Why Do We Need HNSW?](#1-the-problem)
2. [Building Blocks](#2-building-blocks)
3. [HNSW = NSW + Skip List](#3-hnsw-overview)
4. [Key Parameters](#4-key-parameters)
5. [Graph Construction — Step by Step](#5-graph-construction)
6. [Adding a New Vector to an Existing Graph](#6-adding-new-vector)
7. [ANN Search — Finding Nearest Neighbors](#7-ann-search)
8. [Summary Cheat Sheet](#8-summary)

---

## 1. The Problem: Why Do We Need HNSW? {#1-the-problem}

In a RAG system, when a user asks a question, we need to find the most similar document chunks to the query. Each chunk is stored as a **vector embedding** (e.g., a 768-dimensional array of numbers).

**The brute-force approach**: Compare the query vector to *every* stored vector, compute the distance, and return the closest ones.

| Dataset Size | Comparisons Needed | Time at 1M ops/sec |
|---|---|---|
| 1,000 vectors | 1,000 | 1 ms |
| 1,000,000 vectors | 1,000,000 | 1 second |
| 1,000,000,000 vectors | 1,000,000,000 | ~17 minutes |

> [!IMPORTANT]
> HNSW reduces search from **O(N)** to **O(log N)**. For 1 billion vectors, that's ~30 comparisons instead of 1 billion.

---

## 2. Building Blocks {#2-building-blocks}

HNSW combines two ideas. Let's understand each one first.

### 2a. Navigable Small World (NSW) Graph

Think of a social network. Even though billions of people exist, you can reach anyone through roughly 6 connections ("six degrees of separation"). This is the **small world property**.

An NSW graph connects each vector to its nearest neighbors. To search:
1. Start at any node
2. Look at all its neighbors
3. Move to the neighbor **closest** to your query
4. Repeat until no neighbor is closer than the current node (you've hit a **local minimum**)

```mermaid
graph LR
    A((Apple)) --- B((Banana))
    A --- M((Mango))
    B --- M
    A --- H((Honey))
    P((Pepper)) --- W((Wasabi))
    P --- G((Ginger))
    W --- G
    G --- L((Lemon))
    L --- A
```

**Problem**: With a flat (single-layer) graph, greedy search can get **stuck in local minima** — you find a node that's locally closest, but the true nearest neighbor is in a completely different part of the graph.

### 2b. Skip List

A skip list is a multi-level linked list where:
- **Bottom level**: contains ALL elements
- **Each higher level**: contains a random **subset** of elements from the level below
- Searching starts at the top (few elements, big jumps) and refines at the bottom

```
Level 2:  [3] ---------------------------------> [50]
Level 1:  [3] --------> [20] ------------------> [50]
Level 0:  [3] -> [7] -> [20] -> [25] -> [40] -> [50]
```

To find 25: Start at Level 2 → jump from 3 to 50 (too far, go down) → Level 1 → jump from 3 to 20 to 50 (too far, go down) → Level 0 → 20 → 25 ✓

> [!TIP]
> **Key insight**: Upper layers act as "express highways" that skip over large portions of the data. Lower layers act as "local streets" for precise navigation.

---

## 3. HNSW = NSW + Skip List {#3-hnsw-overview}

HNSW fuses both ideas into a **multi-layer navigable small world graph**:

| Property | From NSW | From Skip List |
|---|---|---|
| Graph structure | ✅ Each layer is a navigable graph with nearest-neighbor connections | |
| Greedy search | ✅ "Move to the closest neighbor" | |
| Hierarchy | | ✅ Multiple layers with decreasing node count |
| Search strategy | | ✅ Start at top layer, refine downward |
| Local minima fix | | ✅ Upper layers provide "long-range shortcuts" |

```mermaid
graph TB
    subgraph "Layer 2 (Express Highway)"
        P2((Pepper))
    end
    subgraph "Layer 1 (Regular Highway)"
        P1((Pepper)) --- B1((Banana))
    end
    subgraph "Layer 0 (All Nodes)"
        A0((Apple)) --- B0((Banana))
        A0 --- M0((Mango))
        B0 --- M0
        P0((Pepper)) --- W0((Wasabi))
        P0 --- G0((Ginger))
        W0 --- G0
        A0 --- H0((Honey))
        G0 --- L0((Lemon))
    end
    P2 -.-> P1
    P1 -.-> P0
    B1 -.-> B0
```

**Rules:**
1. **Layer 0** contains ALL nodes
2. Each higher layer contains a **random subset** of the layer below
3. If a node exists on layer L, it also exists on layers L−1, L−2, ..., 0
4. Each layer is an independent NSW graph (nodes connect to their nearest neighbors within that layer)

---

## 4. Key Parameters {#4-key-parameters}

### 4.1 The Three Knobs

| Parameter | What It Controls | Analogy |
|---|---|---|
| **M** | Max neighbors per node per layer | "How many friends each person can have" |
| **efConstruction** | Candidate list size during graph building | "How carefully we pick those friends" |
| **efSearch** | Candidate list size during querying | "How thoroughly we search the city" |

### 4.2 The Layer Assignment Formula

Each node is randomly assigned a max layer using:

$$l = \lfloor -\ln(\text{rand}(0,1)) \times mL \rfloor$$

where **mL = 1 / ln(M)** is a normalization factor.

**Numerical walkthrough** with M = 3, so mL = 1/ln(3) ≈ 0.91:

| Node | rand value (r) | −ln(r) | × mL (×0.91) | floor = Layer |
|---|---|---|---|---|
| Apple | 0.70 | 0.357 | 0.325 | **0** |
| Banana | 0.55 | 0.598 | 0.544 | **0** |
| Mango | 0.80 | 0.223 | 0.203 | **0** |
| Pepper | 0.15 | 1.897 | 1.726 | **1** |
| Wasabi | 0.60 | 0.511 | 0.465 | **0** |
| Ginger | 0.90 | 0.105 | 0.096 | **0** |
| Honey | 0.45 | 0.799 | 0.727 | **0** |
| Lemon | 0.50 | 0.693 | 0.631 | **0** |

> [!NOTE]
> Notice how **most nodes get layer 0** and only **Pepper** gets layer 1. This is by design — the exponential distribution ensures upper layers are sparse "express lanes."

---

## 5. Graph Construction — Step by Step {#5-graph-construction}

### Our Running Example

We represent 8 food items as 2D vectors: **[sweetness, spiciness]**

| Item | Vector | Assigned Layer |
|---|---|---|
| Apple 🍎 | [9, 1] | 0 |
| Banana 🍌 | [8, 0] | 0 |
| Mango 🥭 | [10, 2] | 0 |
| Pepper 🌶️ | [2, 10] | 1 |
| Wasabi 🟢 | [1, 9] | 0 |
| Ginger 🫚 | [3, 7] | 0 |
| Honey 🍯 | [10, 0] | 0 |
| Lemon 🍋 | [5, 4] | 0 |

**Parameters:** M = 3, efConstruction = 8, mL ≈ 0.91

> [!IMPORTANT]
> **Construction IS insertion**. To build an HNSW graph, you simply insert nodes one by one into an empty graph. There's no separate "build" step.

---

### Insert #1: Apple [9, 1] → Layer 0

The graph is empty, so Apple simply becomes the **entry point**.

```
Layer 0:   (Apple)        ← entry point, no connections yet
```

**State**: entry_point = Apple, max_layer = 0

---

### Insert #2: Banana [8, 0] → Layer 0

**Phase 1** — Greedy descent from top to (layer + 1):
- Banana's layer = 0, so we'd descend from max_layer(0) to layer 1 — but 0 < 1, so **this phase is skipped**.

**Phase 2** — Search and connect at layers 0...0:
- **Layer 0**: Search from Apple with ef = efConstruction = 8
  - Only Apple exists. Candidates: {Apple}
  - dist(Banana, Apple) = √((8−9)² + (0−1)²) = √2 ≈ **1.41**
- Connect: **Banana ↔ Apple**

```
Layer 0:   (Apple) ——1.41—— (Banana)
```

---

### Insert #3: Mango [10, 2] → Layer 0

**Phase 2** at Layer 0: Search from Apple with ef=8:

| Candidate | Distance to Mango [10,2] |
|---|---|
| Apple [9,1] | √((10−9)² + (2−1)²) = √2 ≈ **1.41** |
| Banana [8,0] | √((10−8)² + (2−0)²) = √8 ≈ **2.83** |

Select M=3 nearest (only 2 exist): Apple, Banana

Connect: **Mango ↔ Apple**, **Mango ↔ Banana**

```
Layer 0:   (Banana) ——1.41—— (Apple) ——1.41—— (Mango)
              \                                  /
               \——————————2.83————————————————--/
```

---

### Insert #4: Pepper [2, 10] → Layer 1 ⭐

This is the first node assigned to **Layer 1** — the first "express lane" node!

**Phase 1**: Pepper's layer (1) > current max_layer (0), so descent phase is skipped.

**Phase 2**: Search and connect at layers min(0, 1) = 0 down to 0:

**Layer 0**: Search from entry point (Apple) with ef=8. Explore the graph:

| Candidate | Distance to Pepper [2,10] |
|---|---|
| Apple [9,1] | √(49 + 81) = √130 ≈ **11.40** |
| Mango [10,2] | √(64 + 64) = √128 ≈ **11.31** |
| Banana [8,0] | √(36 + 100) = √136 ≈ **11.66** |

Select M=3 nearest: Mango (11.31), Apple (11.40), Banana (11.66)

Connect Pepper to all three at layer 0.

**Update entry point**: Pepper's layer (1) > max_layer (0), so **Pepper becomes the new entry point** and max_layer = 1.

```
Layer 1:   (Pepper)          ← NEW entry point, alone on layer 1

Layer 0:   (Banana)——(Apple)——(Mango)
              \         |        /
               \------(Pepper)--/
```

> [!NOTE]
> Pepper is far from the "sweet" cluster (Apple, Banana, Mango) in vector space, but it still connects to them because they're the only nodes available. These "long-range" connections are actually useful — they provide shortcuts across the graph.

---

### Insert #5: Wasabi [1, 9] → Layer 0

**Phase 1**: Greedy descent from Layer 1 to Layer 1 (since Wasabi's layer = 0, we descend to layer 0+1=1):
- Layer 1: Only Pepper exists. dist(Wasabi, Pepper) = √((1−2)² + (9−10)²) = √2 ≈ **1.41**. Entry point for next layer = Pepper.

**Phase 2** at Layer 0: Search from Pepper with ef=8:
- Start at Pepper. Explore its neighbors: Mango, Apple, Banana
- From Apple, explore its neighbors: Banana (visited), Mango (visited)
- Candidates: {Pepper (1.41), Apple (11.31), Mango (11.40), Banana (11.40)}
- Select M=3 nearest: **Pepper, Apple, Mango**

Connect: **Wasabi ↔ Pepper, Wasabi ↔ Apple, Wasabi ↔ Mango**

---

### Insert #6: Ginger [3, 7] → Layer 0

**Phase 1**: Layer 1 → entry point is Pepper.
- dist(Ginger, Pepper) = √(1 + 9) = √10 ≈ **3.16**. Use Pepper as entry for layer 0.

**Phase 2** at Layer 0: Search from Pepper:
- Pepper's neighbors: Mango, Apple, Banana, Wasabi
- dist(Ginger, Wasabi) = √(4+4) = √8 ≈ **2.83** — closer! Move to Wasabi.
- From Wasabi, check neighbors: Pepper (visited), Apple (visited), Mango (visited)
- Candidates sorted: {**Wasabi (2.83)**, Pepper (3.16), Apple (8.49), Banana (8.60), Mango (8.60)}
- Select M=3 nearest: **Wasabi, Pepper, Apple**

> [!TIP]
> **This insertion demonstrates greedy navigation in action.** We started at Pepper (entry point from layer 1), but the search moved to Wasabi because it was closer to Ginger. The algorithm "walks" through the graph toward the query.

---

### Inserts #7–8: Honey [10,0] and Lemon [5,4]

Same process. After all 8 insertions, the graph looks like:

```mermaid
graph TB
    subgraph "Layer 1"
        P1((Pepper 🌶️))
    end
    subgraph "Layer 0 — All Nodes"
        A((Apple 🍎))
        B((Banana 🍌))
        M((Mango 🥭))
        P((Pepper 🌶️))
        W((Wasabi 🟢))
        G((Ginger 🫚))
        H((Honey 🍯))
        L((Lemon 🍋))
        
        A --- B
        A --- M
        A --- H
        B --- M
        B --- H
        M --- H
        P --- W
        P --- G
        W --- G
        G --- L
        L --- A
        L --- B
    end
    P1 -.->|"same node"| P
```

We can see two natural **clusters** emerging:
- **Sweet cluster**: Apple, Banana, Mango, Honey (top-right in vector space)
- **Spicy cluster**: Pepper, Wasabi, Ginger (top-left in vector space)
- **Lemon** sits in between, bridging the two clusters

---

## 6. Adding a New Vector to an Existing Graph {#6-adding-new-vector}

> [!IMPORTANT]
> Adding a new vector uses the **exact same algorithm** as construction. There is no difference — `insert()` is `insert()`, whether the graph has 0 nodes or 10 million nodes.

### Example: Insert Jalapeño [3, 9]

Let's say Jalapeño gets assigned **layer 1**.

#### Step 1: Assign Layer

```
rand(0,1) = 0.18
l = floor(-ln(0.18) × 0.91) = floor(1.714 × 0.91) = floor(1.56) = 1
```

Jalapeño will exist on **Layer 0 and Layer 1**.

#### Step 2: Phase 1 — Greedy Descent

Jalapeño's layer (1) = max_layer (1), so the descent loop (from max_layer to layer+1 = 2) doesn't execute. No descent needed.

#### Step 3: Phase 2 — Search and Connect

**Layer 1**: Search from Pepper (entry point) with ef=8:
- Only Pepper on layer 1. dist(Jalapeño, Pepper) = √((3−2)² + (9−10)²) = √2 ≈ **1.41**
- Connect: **Jalapeño ↔ Pepper** at Layer 1

**Layer 0**: Search from Pepper with ef=8. The greedy search explores the graph:

| Step | Current Node | Action | 
|---|---|---|
| 1 | Pepper [2,10] | dist = 1.41. Check neighbors: Wasabi, Ginger, Mango, Apple, Banana |
| 2 | Wasabi [1,9] | dist = 2.00. Already explored from Pepper's neighbors |
| 3 | Ginger [3,7] | dist = 2.00. Check its neighbors |

Final candidates sorted by distance to Jalapeño [3,9]:

| Candidate | Distance |
|---|---|
| **Pepper** [2,10] | √2 ≈ **1.41** |
| **Wasabi** [1,9] | √(4+0) = **2.00** |
| **Ginger** [3,7] | √(0+4) = **2.00** |
| Lemon [5,4] | √(4+25) ≈ 5.39 |
| Apple [9,1] | √(36+64) = 10.00 |
| ... | ... |

Select M=3 nearest: **Pepper, Wasabi, Ginger**

Connect Jalapeño to each of these at Layer 0 with bidirectional edges.

#### Step 4: Check Entry Point Update

Jalapeño's layer (1) = max_layer (1), so **no update** needed. (Update only happens when the new node's layer *exceeds* the current max.)

#### Final Graph After Insertion

```mermaid
graph TB
    subgraph "Layer 1"
        P1((Pepper 🌶️)) --- J1((Jalapeño 🫑))
    end
    subgraph "Layer 0"
        A((Apple)) --- B((Banana))
        A --- M((Mango))
        A --- H((Honey))
        P((Pepper)) --- W((Wasabi))
        P --- G((Ginger))
        P --- J((Jalapeño))
        W --- G
        W --- J
        G --- J
        G --- L((Lemon))
        L --- A
    end
    P1 -.-> P
    J1 -.-> J
```

---

## 7. ANN Search — Finding Nearest Neighbors {#7-ann-search}

### Query: "Find the 3 nearest neighbors to Sriracha [4, 8]"

**Parameters**: K = 3 (return top 3), efSearch = 10 (candidate budget)

#### Step 1: Start at the Top Layer

Entry point = **Pepper** on **Layer 1**.

#### Step 2: Greedy Descent (Layer 1 → Layer 0)

At Layer 1, search with ef = 1 (we just want the single closest node to navigate from):

| Node on Layer 1 | Distance to Sriracha [4,8] |
|---|---|
| Pepper [2,10] | √((4−2)² + (8−10)²) = √(4+4) = √8 ≈ **2.83** |
| Jalapeño [3,9] | √((4−3)² + (8−9)²) = √(1+1) = √2 ≈ **1.41** |

Start at Pepper (entry point). Check Pepper's Layer 1 neighbor: Jalapeño.
Jalapeño (1.41) is closer than Pepper (2.83) → move to **Jalapeño**.
No more unvisited neighbors on Layer 1. Best node = **Jalapeño**.

**Descend to Layer 0** using Jalapeño as the starting point.

#### Step 3: Layer 0 — Full Search with ef = 10

Now we do a thorough search on Layer 0 starting from Jalapeño, maintaining up to 10 candidates:

**Iteration 1**: Start with Jalapeño [3,9], dist = 1.41
- Check Jalapeño's neighbors: Pepper, Wasabi, Ginger

| Candidate | Distance to Sriracha [4,8] |
|---|---|
| Jalapeño [3,9] | **1.41** |
| Ginger [3,7] | √(1+1) = **1.41** |
| Pepper [2,10] | **2.83** |
| Wasabi [1,9] | √(9+1) = **3.16** |

**Iteration 2**: Pop closest unvisited = Ginger [3,7], dist = 1.41
- Check Ginger's neighbors: Wasabi (visited), Pepper (visited), Lemon

| Candidate | Distance |
|---|---|
| Ginger [3,7] | **1.41** |
| Jalapeño [3,9] | **1.41** |
| Pepper [2,10] | **2.83** |
| Wasabi [1,9] | **3.16** |
| Lemon [5,4] | √(1+16) ≈ **4.12** |

**Iteration 3**: Pop Pepper [2,10], dist = 2.83
- Check Pepper's unvisited Layer 0 neighbors: Mango, Apple, Banana (depending on connections)
- These are all far away (dist > 8). They're added to candidates but don't improve top results.

**Iteration 4**: Pop Wasabi [1,9], dist = 3.16
- Check Wasabi's unvisited neighbors. All either visited or far away.

**Iterations 5+**: Remaining candidates are farther than our worst result → **STOP**.

#### Step 4: Return Top K = 3 Results

| Rank | Node | Distance | 
|---|---|---|
| 🥇 #1 | **Ginger** [3, 7] | √2 ≈ **1.41** |
| 🥈 #2 | **Jalapeño** [3, 9] | √2 ≈ **1.41** |
| 🥉 #3 | **Pepper** [2, 10] | √8 ≈ **2.83** |

#### Verification: Brute-Force Distances

| Node | Distance to Sriracha [4,8] |
|---|---|
| **Ginger** [3,7] | **1.41** ✅ |
| **Jalapeño** [3,9] | **1.41** ✅ |
| **Pepper** [2,10] | **2.83** ✅ |
| Wasabi [1,9] | 3.16 |
| Lemon [5,4] | 4.12 |
| Mango [10,2] | 8.49 |
| Apple [9,1] | 8.60 |
| Banana [8,0] | 8.94 |
| Honey [10,0] | 10.00 |

> [!TIP]
> In this case, HNSW found the **exact** 3 nearest neighbors while visiting only ~6 nodes out of 9 total. With millions of vectors, the savings are enormous — O(log N) vs O(N).

### Why Is It "Approximate"?

HNSW doesn't guarantee finding the true nearest neighbor because:
1. **Greedy navigation** may lead to a local minimum in the wrong neighborhood
2. **Limited ef budget** means we don't explore every path
3. **Random layer assignments** mean the graph structure varies by seed

In practice, HNSW achieves **>95% recall** (finds the true nearest neighbor 95%+ of the time) with proper parameter tuning.

---

## 8. Summary Cheat Sheet {#8-summary}

### The Three Operations at a Glance

```
┌─────────────────────────────────────────────────────────────────┐
│                     CONSTRUCTION                                │
│  For each vector:                                               │
│    1. Assign random layer: l = floor(-ln(rand) × mL)            │
│    2. Greedy descent from top to l+1 (ef=1)                     │
│    3. At layers l...0: search(ef=efConstruction), connect to M  │
│    4. If l > max_layer: update entry point                      │
│                                                                 │
│  Construction IS repeated insertion. Same algorithm.            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      INSERTION                                  │
│  Identical to construction. Just call insert() on the           │
│  existing graph. Works the same whether graph has 0 or 10M      │
│  nodes.                                                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      ANN SEARCH                                 │
│    1. Start at entry point on top layer                         │
│    2. Greedy descent from top to layer 1 (ef=1)                 │
│    3. At layer 0: search with ef=efSearch candidates            │
│    4. Return top-K from candidate list                          │
└─────────────────────────────────────────────────────────────────┘
```

### Parameter Tuning Guide

| Want to... | Adjust | Trade-off |
|---|---|---|
| Better recall (accuracy) | ↑ efSearch | Slower queries |
| Better graph quality | ↑ efConstruction | Slower build time |
| More connections | ↑ M | More memory, better recall |
| Faster queries | ↓ efSearch | Lower recall |
| Less memory | ↓ M | Fewer connections, may hurt recall |

### How HNSW Fits in RAG

```mermaid
flowchart LR
    A[User Query] --> B[Embedding Model]
    B --> C[Query Vector]
    C --> D["HNSW Index\n(Vector DB)"]
    D --> E[Top-K Similar Chunks]
    E --> F[LLM + Context]
    F --> G[Answer]
```

In a RAG pipeline:
1. Documents are chunked and embedded into vectors
2. Vectors are inserted into an HNSW index (construction)
3. When a query arrives, it's embedded and searched against the index (ANN search)
4. The top-K most similar chunks become the context for the LLM

### C++ Implementation

A complete, runnable C++ implementation is provided at:
[hnsw.cpp](file:///home/midnight/Documents/IIITD/Research/HNSW/hnsw.cpp)

Compile and run:
```bash
g++ -std=c++17 -o hnsw hnsw.cpp -lm
./hnsw
```





The user has 4 specific questions about HNSW that they don't understand from the existing explanation. Let me think through each carefully before writing a supplementary explanation.


efConstruction during insertion
: This is the size of the candidate list maintained during the greedy search when inserting a new node. It controls how many candidates we track during the search phase at each layer when connecting a new node. Higher efConstruction = more candidates explored = better quality connections = better graph quality but slower build time.


Different neighbors on different layers
: YES, absolutely! A node that exists on layer 1 and layer 0 has INDEPENDENT neighbor lists for each layer. Layer 1 connections are formed from the subset of nodes that also exist on layer 1. Layer 0 connections are formed from all nodes.


So a node on layer 1 might connect to one set of neighbors on layer 1 and a completely different (and typically more) neighbors on layer 0.


Pruning function
: When connecting a new node, after finding M nearest neighbors, we add bidirectional edges. But the existing neighbor might already have M connections. So we need to prune—keeping only the M closest connections for that existing node. The heuristic-based pruning from the paper is more sophisticated, avoiding redundant edges if a neighbor can be reached more efficiently through another path, which maintains the diverse connectivity property.


Min


and max-heap usage in search**: During the search process, I'm maintaining a max-heap for the best candidates found so far (where the worst of the best sits at the top for easy comparison) and a min-heap for nodes to explore next (keeping the closest unexplored node accessible). The results heap stays capped at ef, while the candidates heap naturally shrinks as nodes get visited. The search terminates once the closest unvisited candidate becomes farther than the worst result we've found, since that means further exploration won't improve our results.
Now I'm setting up a concrete example with 6 points in 2D space, using M=2 neighbors per layer and efConstruction=3 to keep things manageable while clearly showing how different nodes appear on different layers. P6 [5, 4] → layer 0, so I have a concrete setup where P1 spans all three layers, P2 and P4 appear on layers 0 and 1, and P3, P5, P6 only exist at layer 0. This lets me trace through how efConstruction works during insertions, how the same node maintains different neighbor sets across layers, and what happens when pruning kicks in to enforce the M limit.


Now I'm planning out a comprehensive walkthrough that covers the heap mechanics during search—specifically how the candidate list and visited set grow and shrink as we explore neighbors, when we stop expanding, and what the size constraints actually are. I'm walking through a concrete example with 6 nodes and ef=3, initializing the search with an entry point and tracing how the min-heap tracks candidates while the max-heap maintains the best results found so far, checking at each step whether to continue exploring based on distance comparisons. on layer 0, and F also exists only on layer 0. So layer 1 contains just A and D, while layer 0 has all six points.


Now I'm tracing through the insertion process. When I insert A at layer 1, it becomes the entry point with no connections yet. For B's insertion at layer 0, I descend from layer 1 down to layer 0, finding A as the nearest neighbor at distance 3, then move to layer 0 to continue the search.


At layer 0, I search from A with ef=3, which gives me A as the only result. Since B has no other neighbors available, I connect B to A. After this insertion, A sits alone at layer 1 while A and B are connected at layer 0.


Now inserting C at layer 0 follows the same descent pattern—I start at layer 1 with A, calculate the distance to C as approximately 2.24, and use A as my entry point for the layer 0 search.


At layer 0, I search from A with ef=3. I find B as a neighbor with distance approximately 2.83, which is worse than A's distance but still gets added since I haven't filled my results yet. Now I'm examining B's neighbors to continue the search.


After inserting C, the graph at layer 0 connects A to both B and C, with B and C also connected. Now I'm inserting D at layer 1, which matches the current maximum layer, so there's no descent phase needed—D enters directly at its assigned layer.


At layer 1, only A exists so far, and D connects to it with distance approximately 4.47. Then I'm searching layer 0 from A with the expansion factor, checking A's neighbors B and C to find the nearest candidates for D's connections at that layer. B has distance 2.24, which is less than the worst result, so I add it to both heaps. Then I check C with distance 3, which also qualifies since the results heap isn't full yet. Now I pop B from candidates and continue exploring its neighbors.


After checking B and C's neighbors at layer 0 with no new nodes found, the candidates queue empties. My final results contain the three nearest neighbors: B at 2.24, C at 3.00, and A at 4.47. I select the M=2 closest ones—B and C—and connect D bidirectionally to both at layer 0. This updates the graph structure, adding edges between D and its nearest neighbors while maintaining the existing connections.


Now inserting E at layer 0. I descend from the entry point at layer 1, checking A and D to find the closest node. A is nearer to E at distance 3 compared to D's 4.12, so I use A as my starting point. Then at layer 0, I begin searching from A with ef=3, examining its neighbors B and C. I initialize the search with A in both the candidates and results sets, then start evaluating distances to find E's nearest neighbors. Pop D and check its neighbors—B's distance doesn't qualify since we've already reached our limit of 3 results. Then pop B, but its distance exceeds the worst candidate, so we stop the search. The final nearest neighbors are C, A, and D sorted by distance, and we select the M=2 closest ones (C and A) to connect to E at layer 0.


Now moving to insert F at layer 0—I'm descending from layer 1 by checking distances from A and D, finding that D is closer to F, so I move there and set it as the entry point for the next phase.


In layer 0, I'm searching from D with ef=3, examining its neighbors B and C. I start with D as a candidate, then evaluate B and C based on their distances to F, adding them to the results set when they meet the criteria for the search.


Now I'm continuing the search by popping C from the candidates. I check its neighbors A, B, D, and E, but none of them qualify for addition since their distances don't improve the worst distance in the current results, and the results set is already full at size ef=3.


Next I pop B and continue evaluating, checking whether its distance still satisfies the search conditions.


Once the candidates are exhausted, I have my final results: D at distance 1.41, C at 2.24, and B at 3. I sort these and select the M=2 nearest neighbors, which are D and C. I then connect F bidirectionally to both D and C at layer 0.


Now I'm checking whether D needs pruning by counting its layer 0 neighbors—B, C, and F give it 3 connections, which is within the M_max0 limit of 4, so no pruning is needed. I'm moving on to check C's layer 0 neighbors.


C has 5 neighbors—A, B, D, E, and F—which exceeds the M_max0 limit of 4, so I need to prune. I calculate distances from C to all its neighbors and sort them: E is closest at 1.41, followed by A and F both at 2.24, then B at 2.83, and D furthest at 3.00. I keep the 4 nearest—E, A, F, and B—and remove D from C's neighbor list. This gives me a concrete example of how the pruning mechanism works to maintain the degree constraints.


Now let me verify the final edge list across both layers. After all insertions, layer 0 has edges between A-B, A-C, B-C, B-D, C-D (though D might be removed from C after pruning), A-E, C-E, and D-F.


When F is inserted and C gets pruned, C loses D from its neighbor list, but I need to check whether the pruning function also removes the reverse edge from D's side or if that creates an asymmetry in the adjacency structure. , A)], visited={A}


A's neighbors on layer 1: [D]
dist(Q,D) = √((3-5)²+(4-3)²) = √(4+1) = √5 ≈ 2.24
D is closer, so candidates=[(2.24, D), (3.61, A)], results=[(3.61, A)]
Process D: neighbors [A] already visited
candidates is now empty, layer 1 search done
Layer 0 descent (ef=3):


Start: candidates=[(2.24, D)], results=[(2.24, D)], visited={A, D}
Process D: neighbors [B, F]
dist(Q,B) = √((3-4)²+(4-1)²) = √(1+9) = √10 ≈ 3.16
dist(Q,F) = √((3-4)²+(4-4)²) = √(1+0) = 1
Add both: candidates=[(1, F), (2.24, D), (3.16, B)], results=[(3.16, B), (2.24, D)]
Process F: neighbors [D, C]
D already visited
dist(Q,C) = √((3-2)²+(4-3)²) = √(1+1) = √2 ≈ 1.41
Add C: candidates=[(1.41, C), (1, F), (2.24, D), (3.16, B)], results=[(3.16, B), (2.24, D), (1.41, C)]
candidates size is 4, but ef=3, so we keep the 3 closest in results
Process C: neighbors [A, B, E]
A, B already visited
dist(Q,E) = √((3-1)²+(4-4)²) = √(4+0) = 2
Add E: candidates now has E
We've processed enough candidates (ef limit), return top K=2 from results
Final results:
F (dist 1) and C (dist 1.41) Continuing the layer 0 search, I'm checking D's neighbors B and F. B is at distance 3.16, which doesn't beat the current worst of 2.24, but I add it anyway since the results list isn't full yet. F is much closer at distance 1.00, so it definitely gets added and becomes the new worst candidate at 3.16. Now I'm popping F to explore its neighbors. Now checking C's neighbors in layer 0: A is too far at 3.61, B also doesn't qualify at 3.16, but E comes in at 2.00 which beats the current worst distance of 2.24, so I'm adding E to the results and updating the candidates list.


After removing D to maintain the size limit, I move to iteration 4 where E pops with distance 2.00—checking its neighbors, A is still too far away. Then B pops at 3.16, which exceeds the worst distance of 2.00, so the search terminates and I have my final three nearest neighbors: F, C, and E.


Returning the top 2 as requested gives F(1.00) and C(1.41), which matches the brute force verification perfectly. This example nicely demonstrates how efConstruction works during insertions and how the algorithm maintains different neighbor sets across layers.


Now I'm going to write this up as a comprehensive markdown document that walks through these four key concepts visually and clearly, keeping it focused but thorough.