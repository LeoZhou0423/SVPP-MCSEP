# Single Vehicle Path Planning with Multiple Candidate Start/End Points
import math
import random
import time
import os
import multiprocessing as mp
from typing import List, Tuple, Dict, Optional, Callable, Any
import numpy as np
from dataclasses import dataclass
from functools import partial

# For spatial indexing
from scipy.spatial import KDTree
from sklearn.cluster import AgglomerativeClustering

# For 3D visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 3D Coordinate Structure ---
@dataclass
class Coordinate3D:
    x: float  # Longitude or X coordinate
    y: float  # Latitude or Y coordinate  
    z: float  # Height or Z coordinate
    
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x = x
        self.y = y
        self.z = z
        
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array for spatial indexing"""
        return np.array([self.x, self.y, self.z])
        
    def __repr__(self) -> str:
        return f"Coordinate3D({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"

# --- Fast 3D Distance Calculator (Vectorized Optimization) ---
class FastDistanceCalculator3D:
    def __init__(self, enable_cache: bool = True):
        self.use_cache = enable_cache
        self.distance_cache: Dict[Tuple[int, int], float] = {}
        self._precomputed_matrix = None
        self._points_array = None
    
    def clear_distance_cache(self):
        self.distance_cache.clear()
        self._precomputed_matrix = None
        self._points_array = None
    
    def precompute_distance_matrix(self, points: List[Coordinate3D]) -> np.ndarray:
        """Precompute distance matrix between all points - Vectorized optimized version"""
        n = len(points)
        if n == 0:
            return np.array([])
            
        # Convert to numpy array
        points_array = np.array([[p.x, p.y, p.z] for p in points])
        self._points_array = points_array
        
        # Use broadcasting to compute differences between all point pairs
        diff = points_array[:, np.newaxis, :] - points_array[np.newaxis, :, :]
        
        # Compute Euclidean distances
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
        
        # Fill cache
        if self.use_cache:
            for i in range(n):
                for j in range(n):
                    if i != j:
                        self.distance_cache[(i, j)] = dist_matrix[i, j]
        
        self._precomputed_matrix = dist_matrix
        return dist_matrix
    
    def calculate(self, a: Coordinate3D, b: Coordinate3D) -> float:
        """Calculate 3D Euclidean distance between two points"""
        dx = a.x - b.x
        dy = a.y - b.y
        dz = a.z - b.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def get_distance(self, points: List[Coordinate3D], i: int, j: int) -> float:
        """Distance query - Optimized version"""
        if i == j:
            return 0.0
            
        # First check precomputed matrix
        if self._precomputed_matrix is not None and i < len(self._precomputed_matrix) and j < len(self._precomputed_matrix):
            return self._precomputed_matrix[i, j]
        
        # Then check cache
        if self.use_cache:
            key1 = (i, j)
            key2 = (j, i)
            
            if key1 in self.distance_cache:
                return self.distance_cache[key1]
            if key2 in self.distance_cache:
                return self.distance_cache[key2]
        
        # Finally compute
        dist = self.calculate(points[i], points[j])
        if self.use_cache:
            self.distance_cache[(i, j)] = dist
        return dist
    
    def batch_get_distances(self, points: List[Coordinate3D], indices: List[Tuple[int, int]]) -> List[float]:
        """Batch get distances - Reduce function call overhead"""
        if self._precomputed_matrix is not None:
            return [self._precomputed_matrix[i, j] for i, j in indices]
        
        results = []
        for i, j in indices:
            results.append(self.get_distance(points, i, j))
        return results

# --- KD-Tree Spatial Index (Efficient Nearest Neighbor Search) ---
class FastSpatialIndex3D:
    def __init__(self, points: List[Coordinate3D]):
        """Initialize KD-Tree with 3D points"""
        self.points = points
        self.point_array = np.array([p.to_numpy() for p in points])
        self.tree = KDTree(self.point_array)
    
    def find_k_nearest(self, query_point: Coordinate3D, k: int) -> List[Tuple[int, float]]:
        """Find k nearest neighbors of query point"""
        k = min(k, len(self.points))
        if k <= 0:
            return []
            
        dists, indices = self.tree.query(query_point.to_numpy(), k=k)
        
        # Handle k=1 case (return scalar instead of array)
        if k == 1:
            return [(int(indices), float(dists))]
        
        return [(int(idx), float(dist)) for idx, dist in zip(indices, dists)]
    
    def find_nearest(self, query_point: Coordinate3D) -> Tuple[int, float]:
        """Find nearest neighbor of query point"""
        return self.find_k_nearest(query_point, 1)[0]
    
    def find_within_radius(self, query_point: Coordinate3D, radius: float) -> List[Tuple[int, float]]:
        """Find all points within given radius of query point"""
        indices = self.tree.query_ball_point(query_point.to_numpy(), radius)
        if not indices:
            return []
            
        # Batch compute distances
        query_array = query_point.to_numpy()
        distances = np.linalg.norm(self.point_array[indices] - query_array, axis=1)
        
        return sorted([(int(idx), float(dist)) for idx, dist in zip(indices, distances)], key=lambda x: x[1])

# --- Algorithm Configuration (Optimization Parameters) ---
@dataclass
class FastAlgorithmConfig:
    # Basic parameters - Optimized for convergence speed
    population_size: int = 50  # 减少种群大小
    cooling_rate: float = 0.998  # 更慢的冷却速度
    initial_temperature: float = 5000.0  # 更低的初始温度
    max_iterations: int = 20000  # 减少最大迭代次数
    use_cache: bool = True
    enable_local_search: bool = True
    
    # Initial solution construction
    initial_solution_method: str = "greedy_nn"  # 更快的初始解方法
    
    # ALNS parameters - Optimized performance
    enable_alns: bool = True
    alns_max_iterations: int = 1000  # 减少ALNS迭代
    alns_destroy_rate_min: float = 0.15
    alns_destroy_rate_max: float = 0.3  # 更小的破坏率
    alns_weight_adaptation_rate: float = 0.95
    alns_beta: float = 0.3  # 更积极的反应机制
    
    # Clustering parameters
    enable_clustering: bool = True
    clustering_threshold: float = 20.0  # 更大的阈值减少聚类数量
    clustering_max_depth: int = 1  # 减少聚类深度
    
    # Parallelization parameters
    enable_parallel: bool = True
    parallel_workers: int = None
    
    # Multi-stage optimization - Optimized stage allocation
    enable_multi_stage: bool = True
    stage_1_iterations: int = 2000   # 快速探索
    stage_2_iterations: int = 5000   # 中等优化
    stage_3_iterations: int = 3000   # 精细调整
    
    # Early termination conditions
    enable_early_stop: bool = True
    early_stop_no_improvement: int = 1000  # 无改进时提前终止
    
    # Sampling optimization
    enable_sampling: bool = True
    sampling_ratio: float = 0.3  # 采样比率

# --- Path Planning Result ---
@dataclass
class RouteResult:
    path: List[Coordinate3D]
    total_distance: float
    execution_time_ms: float
    start_index: int
    end_index: int
    optimization_stages: int = 1
    clustering_used: bool = False
    cluster_count: int = 0
    algorithm_used: str = "Hybrid"

# --- Optimized ALNS Operators ---
class FastALNSOperators:
    def __init__(self, distance_calculator: FastDistanceCalculator3D):
        self.dist_calc = distance_calculator
        
    def greedy_nn_construction(self, waypoints: List[Coordinate3D]) -> List[int]:
        """Construct initial solution using greedy nearest neighbor - Optimized version"""
        n = len(waypoints)
        if n <= 2:
            return list(range(n))
        
        # Use spatial index for acceleration
        spatial_index = FastSpatialIndex3D(waypoints)
        
        path = [0] * n
        visited = [False] * n
        
        # Randomly select starting point
        current = random.randint(0, n - 1)
        path[0] = current
        visited[current] = True
        
        # Build path incrementally
        for i in range(1, n):
            # Use spatial index to find nearest neighbors
            nearest = spatial_index.find_k_nearest(waypoints[current], min(10, n - i))
            
            # Find nearest unvisited point
            next_point = current
            for idx, _ in nearest:
                if not visited[idx]:
                    next_point = idx
                    break
            
            # If spatial index fails, fall back to linear search
            if next_point == current:
                for j in range(n):
                    if not visited[j]:
                        next_point = j
                        break
            
            path[i] = next_point
            visited[next_point] = True
            current = next_point
        
        return path
    
    def destroy_random_fast(self, path: List[int], points: List[Coordinate3D], destroy_rate: float) -> Tuple[List[int], List[int]]:
        """Fast random destroy operation"""
        n = len(path)
        destroy_count = max(2, int(n * destroy_rate))
        
        # Use numpy to accelerate random selection
        remove_indices = np.random.choice(n, size=destroy_count, replace=False)
        remove_indices = sorted(remove_indices.tolist(), reverse=True)
        
        removed_points = []
        remaining_path = path.copy()
        
        for idx in remove_indices:
            removed_points.append(remaining_path.pop(idx))
        
        return remaining_path, removed_points
    
    def destroy_worst_fast(self, path: List[int], points: List[Coordinate3D], destroy_rate: float) -> Tuple[List[int], List[int]]:
        """Fast worst edge destroy - Using precomputed distances"""
        n = len(path)
        destroy_count = max(2, int(n * destroy_rate))
        
        # Batch compute edge distances
        edge_distances = []
        for i in range(n - 1):
            dist = self.dist_calc.get_distance(points, path[i], path[i+1])
            edge_distances.append((dist, i))
        
        # Partial sort instead of full sort
        edge_distances.sort(reverse=True, key=lambda x: x[0])
        
        # Select points to remove
        remove_indices_set = set()
        for _, edge_idx in edge_distances[:destroy_count]:
            remove_indices_set.add(path[edge_idx])
            remove_indices_set.add(path[edge_idx + 1])
        
        remove_indices = sorted([path.index(p) for p in remove_indices_set], reverse=True)
        removed_points = []
        remaining_path = path.copy()
        
        for idx in remove_indices:
            removed_points.append(remaining_path.pop(idx))
        
        return remaining_path, removed_points
    
    def repair_greedy_fast(self, partial_path: List[int], removed_points: List[int], 
                          points: List[Coordinate3D]) -> List[int]:
        """Fast greedy repair - Optimized insertion calculation"""
        result = partial_path.copy()
        
        for p in removed_points:
            best_pos = 0
            best_cost = float('inf')
            
            # Batch compute insertion costs
            for i in range(len(result) + 1):
                cost = 0
                if i > 0:
                    cost += self.dist_calc.get_distance(points, result[i-1], p)
                if i < len(result):
                    cost += self.dist_calc.get_distance(points, p, result[i])
                
                if cost < best_cost:
                    best_cost = cost
                    best_pos = i
            
            result.insert(best_pos, p)
        
        return result

# --- Optimized 3D Path Planner ---
class FastRoutePlanner3D:
    def __init__(self, config: FastAlgorithmConfig = None):
        self.config = config if config else FastAlgorithmConfig()
        self.dist_calc = FastDistanceCalculator3D(self.config.use_cache)
        self.alns_operators = FastALNSOperators(self.dist_calc)
        self.spatial_index = None
        random.seed(time.time())
        
    def construct_initial_solution(self, waypoints: List[Coordinate3D]) -> List[int]:
        """Construct initial solution - Choose fastest method based on configuration"""
        if self.config.initial_solution_method == "greedy_nn":
            return self.alns_operators.greedy_nn_construction(waypoints)
        else:
            # Default to fast greedy method
            return self.construct_fast_greedy_path(waypoints)
    
    def construct_fast_greedy_path(self, waypoints: List[Coordinate3D]) -> List[int]:
        """Fast greedy path construction - Using spatial index"""
        n = len(waypoints)
        if n == 0:
            return []
        
        # Build spatial index
        if self.spatial_index is None:
            self.spatial_index = FastSpatialIndex3D(waypoints)
        
        path = [0] * n
        visited = [False] * n
        
        # Randomly select starting point
        current = random.randint(0, n - 1)
        path[0] = current
        visited[current] = True
        
        # Build path incrementally
        for i in range(1, n):
            # Find multiple nearest neighbors to improve probability of finding unvisited points
            k = min(15, n - i + 5)  # 稍微多找一些点
            neighbors = self.spatial_index.find_k_nearest(waypoints[current], k)
            
            # Find nearest unvisited point
            next_point = current
            for idx, _ in neighbors:
                if idx < n and not visited[idx]:
                    next_point = idx
                    break
            
            # Fallback mechanism
            if next_point == current:
                for j in range(n):
                    if not visited[j]:
                        next_point = j
                        break
            
            path[i] = next_point
            visited[next_point] = True
            current = next_point
        
        return path
    
    def calculate_path_distance_fast(self, waypoints: List[Coordinate3D], path_indices: List[int]) -> float:
        """Fast path distance calculation - Using precomputed distances"""
        if len(path_indices) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(path_indices) - 1):
            total += self.dist_calc.get_distance(waypoints, path_indices[i], path_indices[i + 1])
        return total
    
    def two_opt_swap_fast(self, path: List[int], i: int, j: int) -> List[int]:
        """Fast 2-opt swap operation"""
        return path[:i] + path[i:j+1][::-1] + path[j+1:]
    
    def fast_simulated_annealing(self, waypoints: List[Coordinate3D], max_iterations: int = None) -> List[int]:
        """Fast simulated annealing algorithm - Optimized convergence speed"""
        n = len(waypoints)
        if n <= 1:
            return list(range(n))
        
        # Generate initial solution
        current_path = self.construct_initial_solution(waypoints)
        best_path = current_path.copy()
        
        current_dist = self.calculate_path_distance_fast(waypoints, current_path)
        best_dist = current_dist
        
        # Temperature parameters
        temp = self.config.initial_temperature
        cooling_rate = self.config.cooling_rate
        max_iter = max_iterations if max_iterations is not None else self.config.max_iterations
        
        # Early termination tracking
        no_improvement_count = 0
        
        for iteration in range(max_iter):
            # Generate new solution - Using 2-opt neighborhood
            i = random.randint(1, n - 2)
            j = random.randint(i + 1, n - 1)
            new_path = self.two_opt_swap_fast(current_path, i, j)
            
            # Compute new distance
            new_dist = self.calculate_path_distance_fast(waypoints, new_path)
            delta = new_dist - current_dist
            
            # Acceptance criterion
            if delta < 0 or (temp > 1e-9 and random.random() < math.exp(-delta / temp)):
                current_path = new_path
                current_dist = new_dist
                
                if new_dist < best_dist:
                    best_path = new_path.copy()
                    best_dist = new_dist
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
            
            # Early termination check
            if self.config.enable_early_stop and no_improvement_count > self.config.early_stop_no_improvement:
                break
            
            # Cooling
            temp *= cooling_rate
            if temp < 1e-9:
                break
        
        return best_path
    
    def fast_adaptive_large_neighborhood_search(self, waypoints: List[Coordinate3D]) -> List[int]:
        """Fast adaptive large neighborhood search"""
        n = len(waypoints)
        if n <= 1:
            return list(range(n))
        
        # Initial solution
        current_path = self.construct_initial_solution(waypoints)
        current_dist = self.calculate_path_distance_fast(waypoints, current_path)
        best_path = current_path.copy()
        best_dist = current_dist
        
        max_iterations = min(self.config.alns_max_iterations, n * 100)  # Reduce iterations
        
        no_improvement_count = 0
        
        for iteration in range(max_iterations):
            # Adaptive destroy rate
            progress = iteration / max_iterations
            destroy_rate = self.config.alns_destroy_rate_min + \
                          (self.config.alns_destroy_rate_max - self.config.alns_destroy_rate_min) * \
                          (1 - progress)
            
            # Select destroy operator (simplified version)
            if random.random() < 0.7:
                partial_path, removed_points = self.alns_operators.destroy_random_fast(
                    current_path, waypoints, destroy_rate
                )
            else:
                partial_path, removed_points = self.alns_operators.destroy_worst_fast(
                    current_path, waypoints, destroy_rate
                )
            
            # Repair operation
            new_path = self.alns_operators.repair_greedy_fast(
                partial_path, removed_points, waypoints
            )
            
            new_dist = self.calculate_path_distance_fast(waypoints, new_path)
            
            # Simple acceptance criterion
            if new_dist < current_dist:
                current_path = new_path
                current_dist = new_dist
                
                if new_dist < best_dist:
                    best_path = new_path.copy()
                    best_dist = new_dist
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
            
            # Early termination
            if self.config.enable_early_stop and no_improvement_count > self.config.early_stop_no_improvement // 2:
                break
        
        return best_path
    
    def smart_cluster_waypoints(self, waypoints: List[Coordinate3D]) -> List[List[int]]:
        """Smart waypoint clustering - Optimized version"""
        if len(waypoints) <= 8:  # Smaller threshold reduces clustering overhead
            return [list(range(len(waypoints)))]
        
        # 转换为numpy数组
        points_array = np.array([p.to_numpy() for p in waypoints])
        
        # 使用更简单的聚类方法
        n_clusters = max(2, min(8, len(waypoints) // 10))  # Limit number of clusters
        
        try:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='average'
            )
            labels = clustering.fit_predict(points_array)
        except:
            # Return single cluster if clustering fails
            return [list(range(len(waypoints)))]
        
        # 按标签分组
        clusters = []
        for i in range(n_clusters):
            cluster = [j for j, label in enumerate(labels) if label == i]
            if cluster:  # Ensure cluster is not empty
                clusters.append(cluster)
        
        return clusters
    
    def solve_with_smart_clustering(self, waypoints: List[Coordinate3D]) -> Tuple[List[int], int]:
        """Solve problem using smart clustering"""
        clusters = self.smart_cluster_waypoints(waypoints)
        
        if len(clusters) == 1:
            # Single cluster, solve directly
            return self.fast_simulated_annealing(waypoints), 1
        
        # 计算每个簇的代表点（质心）
        representatives = []
        cluster_mappings = []
        
        for cluster in clusters:
            centroid = Coordinate3D(
                sum(waypoints[i].x for i in cluster) / len(cluster),
                sum(waypoints[i].y for i in cluster) / len(cluster),
                sum(waypoints[i].z for i in cluster) / len(cluster)
            )
            representatives.append(centroid)
            cluster_mappings.append(cluster)
        
        # 解决代表点的TSP
        cluster_sequence = self.fast_simulated_annealing(representatives)
        
        # 解决每个簇的内部顺序
        final_order = []
        for cluster_idx in cluster_sequence:
            cluster_indices = cluster_mappings[cluster_idx]
            cluster_points = [waypoints[i] for i in cluster_indices]
            
            if len(cluster_points) <= 8:
                # 小簇使用精确方法
                internal_order = self.solve_small_exact(cluster_points)
            else:
                # 大簇使用快速SA
                internal_order = self.fast_simulated_annealing(cluster_points, 1000)
            
            # 映射回原始索引
            for idx in internal_order:
                final_order.append(cluster_indices[idx])
        
        return final_order, len(clusters)
    
    def solve_small_exact(self, waypoints: List[Coordinate3D]) -> List[int]:
        """Exact method for small-scale problems"""
        n = len(waypoints)
        if n > 8:  # Smaller threshold
            return self.fast_simulated_annealing(waypoints, 500)
        
        from itertools import permutations
        
        indices = list(range(n))
        best_perm = indices.copy()
        best_dist = self.calculate_path_distance_fast(waypoints, indices)
        
        # 枚举所有排列
        for perm in permutations(indices):
            perm_list = list(perm)
            dist = self.calculate_path_distance_fast(waypoints, perm_list)
            if dist < best_dist:
                best_dist = dist
                best_perm = perm_list
        
        return best_perm
    
    def parallel_solve_wrapper(self, args: Tuple[Coordinate3D, Coordinate3D, List[Coordinate3D]]) -> Tuple[List[Coordinate3D], float, int, int]:
        """Parallel solving wrapper"""
        start, end, waypoints = args
        return self.solve_single_start_end(start, end, waypoints)
    
    def solve_single_start_end(self, start: Coordinate3D, end: Coordinate3D, waypoints: List[Coordinate3D]) -> Tuple[List[Coordinate3D], float, int, int]:
        """Solve single start-end pair"""
        if not waypoints:
            # Case with no waypoints
            path = [start, end]
            dist = self.dist_calc.calculate(start, end)
            return path, dist, 0, 0
        
        # 根据问题大小选择策略
        if len(waypoints) <= 8:
            internal_order = self.solve_small_exact(waypoints)
        elif len(waypoints) > 50 and self.config.enable_clustering:
            internal_order, _ = self.solve_with_smart_clustering(waypoints)
        else:
            # 使用快速SA
            internal_order = self.fast_simulated_annealing(waypoints)
            
            # 可选的后优化
            if self.config.enable_local_search and len(waypoints) < 100:
                internal_order = self.fast_local_search(waypoints, internal_order)
        
        # 构建完整路径
        full_path = [start]
        for idx in internal_order:
            full_path.append(waypoints[idx])
        full_path.append(end)
        
        # 计算总距离
        total_dist = self.calculate_path_distance_fast(full_path, list(range(len(full_path))))
        
        return full_path, total_dist, 0, 0
    
    def fast_local_search(self, waypoints: List[Coordinate3D], path: List[int]) -> List[int]:
        """Fast local search - 2-opt optimization"""
        if len(path) < 4:
            return path
        
        improved = True
        current_path = path.copy()
        current_dist = self.calculate_path_distance_fast(waypoints, current_path)
        
        # Limit local search iterations
        max_local_iter = min(100, len(path) * 5)
        
        for _ in range(max_local_iter):
            improved = False
            
            # 随机采样边进行测试
            for _ in range(min(50, len(path))):
                i = random.randint(0, len(path) - 3)
                j = random.randint(i + 2, len(path) - 1)
                
                new_path = self.two_opt_swap_fast(current_path, i, j)
                new_dist = self.calculate_path_distance_fast(waypoints, new_path)
                
                if new_dist < current_dist:
                    current_path = new_path
                    current_dist = new_dist
                    improved = True
                    break
            
            if not improved:
                break
        
        return current_path
    
    def plan_route_fast(self, starts: List[Coordinate3D], 
                       waypoints: List[Coordinate3D], 
                       ends: List[Coordinate3D]) -> RouteResult:
        """Main route planning function - Optimized version"""
        start_time = time.time()
        
        # Precompute distance matrix for acceleration
        if waypoints:
            self.dist_calc.precompute_distance_matrix(waypoints)
        
        best_full_path = []
        best_total_distance = float('inf')
        best_start_idx = 0
        best_end_idx = 0
        
        # Check if using parallel processing
        use_parallel = self.config.enable_parallel and len(starts) * len(ends) > 2
        
        if use_parallel:
            # Prepare parallel parameters
            args_list = [(starts[si], ends[ei], waypoints) 
                        for si in range(len(starts)) 
                        for ei in range(len(ends))]
            
            workers = min(mp.cpu_count(), len(args_list), 8)  # Limit max worker processes
            
            with mp.Pool(processes=workers) as pool:
                results = pool.map(self.parallel_solve_wrapper, args_list)
            
            # Find best result
            for idx, (path, dist, _, _) in enumerate(results):
                si = idx // len(ends)
                ei = idx % len(ends)
                
                if dist < best_total_distance:
                    best_total_distance = dist
                    best_full_path = path
                    best_start_idx = si
                    best_end_idx = ei
        else:
            # Sequential processing
            for si in range(len(starts)):
                for ei in range(len(ends)):
                    path, dist, _, _ = self.solve_single_start_end(starts[si], ends[ei], waypoints)
                    
                    if dist < best_total_distance:
                        best_total_distance = dist
                        best_full_path = path
                        best_start_idx = si
                        best_end_idx = ei
        
        end_time = time.time()
        exec_time = (end_time - start_time) * 1000
        
        return RouteResult(
            path=best_full_path,
            total_distance=best_total_distance,
            execution_time_ms=exec_time,
            start_index=best_start_idx,
            end_index=best_end_idx,
            algorithm_used="FastHybrid"
        )

# ===== Coordinate Generation and Import Functions =====
def generate_random_coordinates_3d(count: int, 
                                 x_min: float, x_max: float,
                                 y_min: float, y_max: float,
                                 z_min: float, z_max: float) -> List[Coordinate3D]:
    """Generate random 3D coordinates"""
    coords = []
    for i in range(count):
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        z = random.uniform(z_min, z_max)
        coords.append(Coordinate3D(x, y, z))
    return coords

def import_coordinates_from_file(file_path: str) -> Tuple[List[Coordinate3D], List[Coordinate3D], List[Coordinate3D]]:
    """Import 3D coordinates from a single file using section markers
    
    The file must contain sections marked by special comments:
    - # START_POINTS: Start points section
    - # WAYPOINTS: Waypoints section  
    - # END_POINTS: End points section
    
    Coordinates should be in x, y, z format, separated by commas, spaces, or tabs.
    All three sections must be present in the file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")
    
    start_points = []
    waypoints = []
    end_points = []
    
    # Track which sections have been found
    sections_found = {"START_POINTS": False, "WAYPOINTS": False, "END_POINTS": False}
    current_section = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            
            # Check for section markers
            if line.upper() == "# START_POINTS":
                current_section = "START_POINTS"
                sections_found[current_section] = True
                continue
            elif line.upper() == "# WAYPOINTS":
                current_section = "WAYPOINTS"
                sections_found[current_section] = True
                continue
            elif line.upper() == "# END_POINTS":
                current_section = "END_POINTS"
                sections_found[current_section] = True
                continue
            
            # Skip other comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Check if we're in a valid section
            if current_section is None:
                raise ValueError(f"第{line_idx}行包含坐标但未找到有效的分段标记。请确保文件包含# START_POINTS、# WAYPOINTS和# END_POINTS标记。")
            
            # Try different delimiters (comma, space, tab)
            valid_coordinate = False
            for delimiter in [',', ' ', '\t']:
                parts = line.split(delimiter)
                # Remove empty strings (if any)
                parts = [p.strip() for p in parts if p.strip()]
                
                if len(parts) >= 3:
                    try:
                        # 提取x, y, z坐标
                        x = float(parts[0])
                        y = float(parts[1])
                        z = float(parts[2])
                        coord = Coordinate3D(x, y, z)
                        
                        # Add to corresponding section
                        if current_section == "START_POINTS":
                            start_points.append(coord)
                        elif current_section == "WAYPOINTS":
                            waypoints.append(coord)
                        elif current_section == "END_POINTS":
                            end_points.append(coord)
                        
                        valid_coordinate = True
                        break
                    except ValueError:
                        continue
            
            if not valid_coordinate:
                # If no valid delimiter found
                raise ValueError(f"第{line_idx}行坐标格式无效: {line}")
    
    # Verify all required sections were found
    if not all(sections_found.values()):
        missing_sections = [f"#{section}" for section, found in sections_found.items() if not found]
        raise ValueError(f"缺少必需的分段标记: {', '.join(missing_sections)}。请明确定义所有点类型。")
    
    # Verify at least one start and one end point
    if not start_points:
        raise ValueError("在START_POINTS部分未找到有效的起始点坐标。")
    
    if not end_points:
        raise ValueError("在END_POINTS部分未找到有效的终点坐标。")
    
    return start_points, waypoints, end_points

def print_coordinates_3d(coords: List[Coordinate3D], title: str):
    """Print 3D coordinates"""
    print(f"===== {title} =====")
    for i, coord in enumerate(coords):
        print(f"{i+1}: X={coord.x:.6f}, Y={coord.y:.6f}, Z={coord.z:.6f}")

def print_result_3d(result: RouteResult):
    """Print 3D route planning results"""
    print("\n===== 3D Route Planning Results =====")
    print(f"Total Distance: {result.total_distance:.2f} units")
    print(f"Computation Time: {result.execution_time_ms:.0f} ms")
    print(f"Algorithm Used: {result.algorithm_used}")
    print(f"Optimization Stages: {result.optimization_stages}")
    if result.clustering_used:
        print(f"Clustering Applied: Yes (total {result.cluster_count} clusters)")
    else:
        print(f"Clustering Applied: No")
    print(f"Path Sequence ({len(result.path)} points):")
    
    for i, point in enumerate(result.path):
        print(f"  {i+1}. X={point.x:.6f}, Y={point.y:.6f}, Z={point.z:.6f}")

def plot_route_3d(result: RouteResult, starts: List[Coordinate3D], 
                 waypoints: List[Coordinate3D], ends: List[Coordinate3D]):
    """Plot 3D route planning results using matplotlib"""
    try:
        # Create figure and 3D axis
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract path coordinates
        path_x = [p.x for p in result.path]
        path_y = [p.y for p in result.path]
        path_z = [p.z for p in result.path]
        
        # Separate selected and unselected start points
        selected_start = result.path[0]
        unselected_starts = [p for p in starts if p != selected_start]
        
        # Plot unselected start points (blue circles)
        if unselected_starts:
            unselected_start_x = [p.x for p in unselected_starts]
            unselected_start_y = [p.y for p in unselected_starts]
            unselected_start_z = [p.z for p in unselected_starts]
            ax.scatter(unselected_start_x, unselected_start_y, unselected_start_z, 
                      c='blue', s=120, marker='o', label='Start Points', alpha=0.8)
        
        # Plot waypoints (green squares)
        if waypoints:
            waypoint_x = [p.x for p in waypoints]
            waypoint_y = [p.y for p in waypoints]
            waypoint_z = [p.z for p in waypoints]
            ax.scatter(waypoint_x, waypoint_y, waypoint_z, c='green', s=80, marker='s', label='Waypoint', alpha=0.7)
        
        # Separate selected and unselected end points
        selected_end = result.path[-1]
        unselected_ends = [p for p in ends if p != selected_end]
        
        # Plot unselected end points (red triangles)
        if unselected_ends:
            unselected_end_x = [p.x for p in unselected_ends]
            unselected_end_y = [p.y for p in unselected_ends]
            unselected_end_z = [p.z for p in unselected_ends]
            ax.scatter(unselected_end_x, unselected_end_y, unselected_end_z, 
                      c='red', s=100, marker='^', label='End Points', alpha=0.8)
        
        # Plot optimal path (black line with arrows)
        ax.plot(path_x, path_y, path_z, 'k-', linewidth=3, alpha=0.8, label='Optimal Path')
        
        # Add arrows to indicate direction (reduce arrow count for performance)
        arrow_step = max(1, len(path_x) // 10)  # Show one arrow every 10 points
        for i in range(0, len(path_x) - 1, arrow_step):
            if i + 1 < len(path_x):
                ax.quiver(
                    path_x[i], path_y[i], path_z[i],
                    path_x[i+1] - path_x[i], path_y[i+1] - path_y[i], path_z[i+1] - path_z[i],
                    length=0.8, color='red', arrow_length_ratio=0.2, linewidth=2, alpha=0.7
                )
        
        # Mark selected start and end points (only show these, not overlapping with unselected)
        ax.scatter([selected_start.x], [selected_start.y], [selected_start.z], 
                  c='cyan', s=200, marker='o', edgecolors='black', linewidths=3, label='Selected Start')
        ax.scatter([selected_end.x], [selected_end.y], [selected_end.z], 
                  c='magenta', s=200, marker='^', edgecolors='black', linewidths=3, label='Selected End')
        
        # Add labels
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_zlabel('Z Coordinate', fontsize=12)
        ax.set_title('3D Path Planning Visualization', fontsize=14, fontweight='bold')      
        # Add legend
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=10)    
        # Add grid for better visualization
        ax.grid(True, linestyle='--', alpha=0.6)   
        # Set equal scaling
        all_x = path_x.copy()
        all_y = path_y.copy()
        all_z = path_z.copy()
        
        # Add unselected start points if any
        if unselected_starts:
            all_x.extend(unselected_start_x)
            all_y.extend(unselected_start_y)
            all_z.extend(unselected_start_z)
        
        # Add unselected end points if any
        if unselected_ends:
            all_x.extend(unselected_end_x)
            all_y.extend(unselected_end_y)
            all_z.extend(unselected_end_z)
        
        if waypoints:
            all_x += waypoint_x
            all_y += waypoint_y
            all_z += waypoint_z
        
        max_range = max(
            max(all_x) - min(all_x),
            max(all_y) - min(all_y),
            max(all_z) - min(all_z)
        )
        mid_x = (max(all_x) + min(all_x)) / 2
        mid_y = (max(all_y) + min(all_y)) / 2
        mid_z = (max(all_z) + min(all_z)) / 2
        
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        # Add distance text
        plt.figtext(0.5, 0.02, f'Total Distance: {result.total_distance:.2f} units | Computation Time: {result.execution_time_ms:.0f} ms', 
                   ha='center', fontsize=11, bbox=dict(facecolor='lightblue', alpha=0.8))
        
        # Show figure
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        plt.show()
        
    except Exception as e:
        print(f"Error plotting 3D visualization: {e}")
        print("Please make sure matplotlib is properly installed.")

# Performance test function removed as requested
    


# ===== Main Function =====
def main():
    """Main function with optimized performance"""
    # Configure optimization parameters
    config = FastAlgorithmConfig(
        cooling_rate=0.998,
        initial_temperature=5000.0,
        max_iterations=15000,
        
        # ALNS parameters
        enable_alns=True,
        alns_max_iterations=800,
        
        # Clustering parameters
        enable_clustering=True,
        clustering_threshold=25.0,
        
        # Parallelization
        enable_parallel=True,
        
        # Early termination
        enable_early_stop=True,
        early_stop_no_improvement=500
    )
    
    print("Fast 3D Path Planner")
    print("=" * 50)
    print("Select data input method:")
    print("1. Generate random coordinates")
    print("2. Import coordinates from file")
    
    try:
        input_method = input("Enter selection (1 or 2): ").strip()
        
        if input_method == '1':
            num_starts = int(input("Enter number of start points: "))
            num_waypoints = int(input("Enter number of waypoints: "))
            num_ends = int(input("Enter number of end points: "))
            
            if num_starts <= 0 or num_waypoints < 0 or num_ends <= 0:
                print("Input numbers must be positive integers!")
                return
            
            # Generate random 3D coordinates
            print("Generating random coordinates...")
            starts = generate_random_coordinates_3d(num_starts, 0, 100, 0, 100, 0, 50)
            waypoints = generate_random_coordinates_3d(num_waypoints, 0, 100, 0, 100, 0, 50)
            ends = generate_random_coordinates_3d(num_ends, 0, 100, 0, 100, 0, 50)
            
        elif input_method == '2':
            print("\n--- Import Coordinates ---")
            print("Note: File should contain sections separated by comment markers:")
            print("      # START_POINTS - Start points")
            print("      # WAYPOINTS - Waypoints") 
            print("      # END_POINTS - End points")
            print("      Each coordinate line should contain x, y, z separated by commas, spaces or tabs")
            
            coordinate_file = input("Enter coordinate file path: ").strip()
            if coordinate_file:
                try:
                    starts, waypoints, ends = import_coordinates_from_file(coordinate_file)
                    print(f"\nSuccessfully imported:")
                    print(f"  - {len(starts)} start points")
                    print(f"  - {len(waypoints)} waypoints")
                    print(f"  - {len(ends)} end points")
                except (FileNotFoundError, ValueError) as e:
                    print(f"Error importing coordinates: {e}")
                    return
            else:
                print("No file provided, using default values")
                starts = [Coordinate3D(0, 0, 0)]
                waypoints = generate_random_coordinates_3d(15, 0, 100, 0, 100, 0, 50)
                ends = [Coordinate3D(100, 100, 50)]
                
        else:
            print("Invalid selection! Please enter 1 or 2.")
            return
            
    except ValueError:
        print("Please enter valid integers!")
        return
    
    # Print all points information
    print("\n" + "="*50)
    print_coordinates_3d(starts, "Candidate Start Points")
    if waypoints:
        print_coordinates_3d(waypoints, "Waypoints")
    print_coordinates_3d(ends, "Candidate End Points")
    
    # Execute fast path planning
    planner = FastRoutePlanner3D(config)
    
    print("\nStarting fast path planning...")
    start_time = time.time()
    result = planner.plan_route_fast(starts, waypoints, ends)
    end_time = time.time()
    
    total_time = end_time - start_time
    print(f"\nPlanning completed! Total time: {total_time:.2f} seconds")
    
    # Print results
    print_result_3d(result)
    
    # Display detailed path by default
    print("\nDetailed path order:")
    for i, point in enumerate(result.path):
        point_type = ""
        if i == 0:
            point_type = " [Start]"
        elif i == len(result.path) - 1:
            point_type = " [End]"
        elif point in waypoints:
            point_type = " [Waypoint]"
        
        print(f"  {i+1:2d}. X={point.x:7.2f}, Y={point.y:7.2f}, Z={point.z:6.2f}{point_type}")
    
    # Generate 3D visualization by default
    print("Generating 3D visualization...")
    plot_route_3d(result, starts, waypoints, ends)
    
    # Performance information (if applicable)
    if len(waypoints) > 20:
        print(f"\nPerformance note: Processed {len(waypoints)} points in {total_time:.2f} seconds")
        if total_time > 10:
            print("Consider reducing the number of waypoints or using the clustering option to further improve speed.")
        else:
            print("Excellent performance!")

if __name__ == "__main__":
    main()
