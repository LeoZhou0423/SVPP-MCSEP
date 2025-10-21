您指出的问题非常准确！以下是修改后的完整数学建模文档：

# Mathematical Model for Single Vehicle Path Planning with Multiple Candidate Start/End Points

## Problem Definition

The problem addressed in this project is a variant of the Traveling Salesman Problem (TSP) with the following characteristics:

1. 3D spatial coordinates for all points
2. Multiple candidate start points
3. Multiple candidate end points  
4. Compulsory intermediate waypoints
5. Optimization objective: Minimize total travel distance

## Mathematical Formulation

### Sets and Variables

Let us define the following sets and variables:

- **Sets**:
  - $S = \{s_1, s_2, \ldots, s_m\}$: Set of candidate start points
  - $E = \{e_1, e_2, \ldots, e_k\}$: Set of candidate end points
  - $W = \{w_1, w_2, \ldots, w_n\}$: Set of required waypoints
  - $P = S \cup W \cup E$: Complete set of all points

- **Parameters**:
  - $p_i = (x_i, y_i, z_i) \in \mathbb{R}^3$: 3D coordinates of point $i$
  - $d_{ij}$: Euclidean distance between points $i$ and $j$

- **Decision Variables**:
  - $s \in S$: Selected start point
  - $e \in E$: Selected end point  
  - $\sigma \in \Pi(W)$: Permutation of waypoints, where $\Pi(W)$ denotes the set of all permutations of $W$
  - $\pi = (s, w_{\sigma(1)}, w_{\sigma(2)}, \ldots, w_{\sigma(n)}, e)$: Complete path sequence

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

## Hybrid Optimization Approach

The implemented solution uses a hybrid algorithm combining several optimization techniques:

### 1. Greedy Construction Heuristic

```latex
\begin{algorithmic}
\Function{GreedyConstruction}{$S, W, E$}
    \State $s \gets \text{RandomSelect}(S)$
    \State $e \gets \text{RandomSelect}(E)$
    \State $\text{unvisited} \gets W$
    \State $\text{path} \gets [s]$
    \State $\text{current} \gets s$
    \While{$\text{unvisited} \neq \emptyset$}
        \State $w^* \gets \arg\min_{w \in \text{unvisited}} d(\text{current}, w)$
        \State $\text{path}.\text{append}(w^*)$
        \State $\text{unvisited}.\text{remove}(w^*)$
        \State $\text{current} \gets w^*$
    \EndWhile
    \State $\text{path}.\text{append}(e)$
    \State \Return $\text{path}$
\EndFunction
\end{algorithmic}
```

### 2. Simulated Annealing

Simulated annealing is applied with the following formal components:

- **State Space**: All permutations $\sigma \in \Pi(W)$
- **Energy Function**: 
  $$E(\sigma) = d(s, w_{\sigma(1)}) + \sum_{i=1}^{n-1} d(w_{\sigma(i)}, w_{\sigma(i+1)}) + d(w_{\sigma(n)}, e)$$
- **Neighborhood Operator**: 2-opt move
  $$N(\sigma, i, j) = (w_{\sigma(1)}, \ldots, w_{\sigma(i-1)}, w_{\sigma(j)}, w_{\sigma(j-1)}, \ldots, w_{\sigma(i)}, w_{\sigma(j+1)}, \ldots, w_{\sigma(n)})$$
  where $1 \leq i < j \leq n$
- **Temperature Schedule**: 
  $$T_{k+1} = \alpha \cdot T_k, \quad \alpha = 0.995, \quad T_0 = 10000$$
- **Acceptance Probability**:
  $$P_{\text{accept}} = \begin{cases} 
  1 & \text{if } \Delta E < 0 \\
  \exp(-\Delta E / T) & \text{otherwise}
  \end{cases}$$

### 3. 2-opt Local Search

```latex
\begin{algorithmic}
\Function{TwoOptOptimize}{$\sigma$}
    \State $\text{improved} \gets \text{True}$
    \While{$\text{improved}$}
        \State $\text{improved} \gets \text{False}$
        \For{$i = 1$ to $n-1$}
            \For{$j = i+1$ to $n$}
                \State $\sigma' \gets \text{TwoOptSwap}(\sigma, i, j)$
                \State $\Delta E \gets E(\sigma') - E(\sigma)$
                \If{$\Delta E < 0$}
                    \State $\sigma \gets \sigma'$
                    \State $\text{improved} \gets \text{True}$
                \EndIf
            \EndFor
        \EndFor
    \EndWhile
    \State \Return $\sigma$
\EndFunction
\end{algorithmic}
```

### 4. Exact Solution for Small Instances

For $n \leq 12$, we employ complete enumeration:

```latex
\begin{algorithmic}
\Function{ExactTSP}{$W$}
    \State $\sigma^* \gets \text{identity permutation}$
    \State $E^* \gets E(\sigma^*)$
    \For{$\text{each } \sigma \in \Pi(W)$}
        \State $E_{\text{current}} \gets E(\sigma)$
        \If{$E_{\text{current}} < E^*$}
            \State $\sigma^* \gets \sigma$
            \State $E^* \gets E_{\text{current}}$
        \EndIf
    \EndFor
    \State \Return $\sigma^*$
\EndFunction
\end{algorithmic}
```

## Complete Algorithm Framework

```latex
\begin{algorithmic}
\Function{SVPP-MCSEP}{$S, W, E$}
    \State $\text{best\_path} \gets \emptyset$
    \State $\text{best\_distance} \gets \infty$
    
    \For{$s \in S$}
        \For{$e \in E$}
            \If{$|W| \leq 12$}
                \State $\sigma \gets \text{ExactTSP}(W)$
            \Else
                \State $\sigma \gets \text{GreedyConstruction}(\{s\}, W, \{e\})$
                \State $\sigma \gets \text{SimulatedAnnealing}(\sigma, W, s, e)$
                \State $\sigma \gets \text{TwoOptOptimize}(\sigma)$
            \EndIf
            
            \State $\text{path} \gets (s, w_{\sigma(1)}, \ldots, w_{\sigma(n)}, e)$
            \State $\text{distance} \gets E(\sigma)$
            
            \If{$\text{distance} < \text{best\_distance}$}
                \State $\text{best\_path} \gets \text{path}$
                \State $\text{best\_distance} \gets \text{distance}$
            \EndIf
        \EndFor
    \EndFor
    
    \State \Return $\text{best\_path}, \text{best\_distance}$
\EndFunction
\end{algorithmic}
```

## Complexity Analysis

### Time Complexity
- **Exact Algorithm**: $O(m \cdot k \cdot n!)$
- **Greedy Construction**: $O(m \cdot k \cdot n^2)$  
- **Simulated Annealing**: $O(m \cdot k \cdot I \cdot n^2)$ where $I$ is iteration count
- **2-opt Local Search**: $O(m \cdot k \cdot n^3)$ in worst case

### Space Complexity
- **Distance Cache**: $O((m + n + k)^2)$
- **Algorithm State**: $O(n)$ for current permutation

## Performance Guarantees

- **Optimality**: For $n \leq 12$, the algorithm guarantees global optimality
- **Approximation**: For $n > 12$, the hybrid approach provides high-quality approximate solutions
- **Completeness**: The algorithm always returns a feasible solution satisfying all constraints

## Conclusion

This mathematical model provides a rigorous foundation for the single vehicle path planning problem with multiple candidate start/end points. The hybrid optimization strategy effectively balances solution quality and computational efficiency, making it suitable for both small-scale exact solutions and large-scale practical applications.