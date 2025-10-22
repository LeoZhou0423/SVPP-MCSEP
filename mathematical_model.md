# Mathematical Model for Single Vehicle Path Planning with Multiple Candidate Start/End Points

## Problem Definition

The problem addressed in this project is an extended variant of the Traveling Salesman Problem (TSP) with the following characteristics:

1. 3D spatial coordinates for all points
2. Multiple candidate start points
3. Multiple candidate end points  
4. Compulsory intermediate waypoints
5. Optimization objective: Minimize total travel distance
6. Scalable optimization strategies based on problem size

## Mathematical Formulation

### Sets and Variables

Let us define the following sets and variables:

- **Sets**:
  - $S = \{s_1, s_2, \ldots, s_m\}$: Set of candidate start points
  - $E = \{e_1, e_2, \ldots, e_k\}$: Set of candidate end points
  - $W = \{w_1, w_2, \ldots, w_n\}$: Set of required waypoints
  - $P = S \cup W \cup E$: Complete set of all points
  - $C = \{C_1, C_2, \ldots, C_l\}$: Set of clusters (for large-scale problems)

- **Parameters**:
  - $p_i = (x_i, y_i, z_i) \in \mathbb{R}^3$: 3D coordinates of point $i$
  - $d_{ij}$: Euclidean distance between points $i$ and $j$
  - $\tau_{\text{exact}} = 8$: Threshold for exact solution
  - $\tau_{\text{cluster}} = 50$: Threshold for clustering

- **Decision Variables**:
  - $s \in S$: Selected start point
  - $e \in E$: Selected end point  
  - $\sigma \in \Pi(W)$: Permutation of waypoints, where $\Pi(W)$ denotes the set of all permutations of $W$
  - $\pi = (s, w_{\sigma(1)}, w_{\sigma(2)}, \ldots, w_{\sigma(n)}, e)$: Complete path sequence
  - $\gamma \in \Pi(C)$: Permutation of clusters

### Distance Calculation

The Euclidean distance between two 3D points $i$ and $j$ is calculated as:

$$d_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2 + (z_i - z_j)^2}$$

### Objective Function

The primary objective is to minimize the total travel distance of the route:

$$\min_{s \in S, e \in E, \sigma \in \Pi(W)} \left[ d(s, w_{\sigma(1)}) + \sum_{i=1}^{n-1} d(w_{\sigma(i)}, w_{\sigma(i+1)}) + d(w_{\sigma(n)}, e) \right]$$

### Constraints

1. **Start Point Selection**: 
   $$s \in S$$

2. **End Point Selection**:
   $$e \in E$$

3. **Waypoint Coverage**:
   $$\forall w \in W, \exists i \in \{1, 2, \ldots, n\} : \pi_{i+1} = w$$

4. **Unique Visit Constraint**:
   $$\pi_i \neq \pi_j \quad \forall i \neq j, \quad i,j \in \{2, 3, \ldots, n+1\}$$

5. **Path Structure**:
   $$\pi_1 = s, \quad \pi_{n+2} = e, \quad \{\pi_2, \pi_3, \ldots, \pi_{n+1}\} = W$$

## Hybrid Optimization Framework

### 1. Problem Size-Based Strategy Selection

The algorithm dynamically selects optimization strategies based on problem size:

- **Small-scale** ($n \leq \tau_{\text{exact}}$):
  $$\text{Strategy} = \text{ExactEnumeration}$$

- **Medium-scale** ($\tau_{\text{exact}} < n \leq \tau_{\text{cluster}}$):
  $$\text{Strategy} = \text{FastSimulatedAnnealing} + \text{LocalSearch}$$

- **Large-scale** ($n > \tau_{\text{cluster}}$):
  $$\text{Strategy} = \text{Clustering} + \text{HierarchicalOptimization}$$

### 2. Smart Clustering for Large-Scale Problems

For large-scale instances ($n > \tau_{\text{cluster}}$), we employ hierarchical clustering:

#### Cluster Formation
Let $X = \{x_1, x_2, \ldots, x_n\}$ be the set of waypoint coordinates.

**Agglomerative Clustering**:
$$C = \text{AgglomerativeClustering}(X, n_{\text{clusters}})$$

where $n_{\text{clusters}} = \max\left(2, \min\left(8, \lfloor \frac{n}{10} \rfloor\right)\right)$

#### Representative Points
For each cluster $C_i$, compute centroid:
$$\mu_i = \frac{1}{|C_i|} \sum_{x \in C_i} x$$

#### Two-Level Optimization
1. **Inter-cluster optimization**:
   $$\gamma^* = \arg\min_{\gamma \in \Pi(C)} \left[ d(s, \mu_{\gamma(1)}) + \sum_{i=1}^{l-1} d(\mu_{\gamma(i)}, \mu_{\gamma(i+1)}) + d(\mu_{\gamma(l)}, e) \right]$$

2. **Intra-cluster optimization**:
   For each cluster $C_{\gamma^*(i)}$, solve:
   $$\sigma_i^* = \arg\min_{\sigma \in \Pi(C_{\gamma^*(i)})} \sum_{j=1}^{|C_i|-1} d(w_{\sigma(j)}, w_{\sigma(j+1)})$$

### 3. Fast Simulated Annealing

For medium-scale problems, we use an optimized simulated annealing:

- **State Space**: All permutations $\sigma \in \Pi(W)$
- **Energy Function**: 
  $$E(\sigma) = d(s, w_{\sigma(1)}) + \sum_{i=1}^{n-1} d(w_{\sigma(i)}, w_{\sigma(i+1)}) + d(w_{\sigma(n)}, e)$$
- **Neighborhood Operator**: 2-opt move
  $$N(\sigma, i, j) = (w_{\sigma(1)}, \ldots, w_{\sigma(i-1)}, w_{\sigma(j)}, w_{\sigma(j-1)}, \ldots, w_{\sigma(i)}, w_{\sigma(j+1)}, \ldots, w_{\sigma(n)})$$
- **Temperature Schedule**: 
  $$T_{k+1} = \alpha \cdot T_k, \quad \alpha = 0.998, \quad T_0 = 5000$$
- **Acceptance Probability**:
  $$P_{\text{accept}} = \begin{cases} 
  1 & \text{if } \Delta E < 0 \\
  \exp(-\Delta E / T) & \text{otherwise}
  \end{cases}$$

### 4. Adaptive Large Neighborhood Search (ALNS)

For enhanced local search, we implement ALNS:

#### Destroy Operators
- **Random Destroy**:
  $$D_{\text{random}}(\sigma, \rho) = \text{remove } \lfloor \rho \cdot n \rfloor \text{ random points}$$
- **Worst-edge Destroy**:
  $$D_{\text{worst}}(\sigma, \rho) = \text{remove points incident to worst } \lfloor \rho \cdot n \rfloor \text{ edges}$$

#### Repair Operator
- **Greedy Repair**:
  $$R_{\text{greedy}}(\sigma_{\text{partial}}, P_{\text{removed}}) = \text{insert points at minimum-cost positions}$$

#### Adaptive Mechanism
Destroy rate adapts during search:
$$\rho(t) = \rho_{\min} + (\rho_{\max} - \rho_{\min}) \cdot (1 - \frac{t}{T_{\max}})$$

### 5. Exact Solution for Small Instances

For $n \leq \tau_{\text{exact}}$, we employ complete enumeration:

```latex
\begin{algorithmic}
\Function{ExactTSP}{$W, s, e$}
    \State $\sigma^* \gets \text{identity permutation}$
    \State $E^* \gets E(\sigma^*)$
    \For{$\text{each } \sigma \in \Pi(W)$}
        \State $E_{\text{current}} \gets d(s, w_{\sigma(1)}) + \sum_{i=1}^{n-1} d(w_{\sigma(i)}, w_{\sigma(i+1)}) + d(w_{\sigma(n)}, e)$
        \If{$E_{\text{current}} < E^*$}
            \State $\sigma^* \gets \sigma$
            \State $E^* \gets E_{\text{current}}$
        \EndIf
    \EndFor
    \State \Return $\sigma^*$
\EndFunction
\end{algorithmic}