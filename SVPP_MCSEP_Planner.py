#Single Vehicle Path Planning with Multiple Candidate Start/End Points
import math
import random
import time
import os
from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass

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

# --- 3D Distance Calculator (with caching support) ---
class DistanceCalculator3D:
    def __init__(self, enable_cache: bool = True):
        self.use_cache = enable_cache
        self.distance_cache: Dict[Tuple[int, int], float] = {}
    
    def clear_distance_cache(self):
        self.distance_cache.clear()
    
    def calculate(self, a: Coordinate3D, b: Coordinate3D) -> float:
        """Calculate 3D Euclidean distance between two points"""
        dx = a.x - b.x
        dy = a.y - b.y
        dz = a.z - b.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def get_distance(self, points: List[Coordinate3D], i: int, j: int) -> float:
        """Distance query with caching"""
        if not self.use_cache:
            return self.calculate(points[i], points[j])
        
        key1 = (i, j)
        key2 = (j, i)
        
        if key1 in self.distance_cache:
            return self.distance_cache[key1]
        if key2 in self.distance_cache:
            return self.distance_cache[key2]
        
        dist = self.calculate(points[i], points[j])
        self.distance_cache[key1] = dist
        return dist

# --- Algorithm Configuration ---
@dataclass
class AlgorithmConfig:
    population_size: int = 100
    cooling_rate: float = 0.995
    initial_temperature: float = 10000.0
    max_iterations: int = 100000
    use_cache: bool = True
    enable_local_search: bool = True

# --- Route Planning Result ---
@dataclass
class RouteResult:
    path: List[Coordinate3D]
    total_distance: float
    execution_time_ms: float
    start_index: int
    end_index: int

# --- 3D Route Planner ---
class RoutePlanner3D:
    def __init__(self, config: AlgorithmConfig = None):
        self.config = config if config else AlgorithmConfig()
        self.dist_calc = DistanceCalculator3D(self.config.use_cache)
        random.seed(time.time())
    
    def calculate_path_distance(self, path: List[Coordinate3D]) -> float:
        """Calculate total path distance"""
        if len(path) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(path) - 1):
            total += self.dist_calc.calculate(path[i], path[i + 1])
        return total
    
    def construct_greedy_path(self, waypoints: List[Coordinate3D]) -> List[int]:
        """Construct greedy path: start from random point, choose nearest unvisited point each time"""
        n = len(waypoints)
        if n == 0:
            return []
        
        path = [0] * n
        visited = [False] * n
        
        # Randomly select starting point
        current = random.randint(0, n - 1)
        path[0] = current
        visited[current] = True
        
        # Build path step by step
        for i in range(1, n):
            best_dist = float('inf')
            next_point = current
            
            # Find nearest unvisited point
            for j in range(n):
                if not visited[j]:
                    d = self.dist_calc.calculate(waypoints[current], waypoints[j])
                    if d < best_dist:
                        best_dist = d
                        next_point = j
            
            if next_point != current:
                path[i] = next_point
                visited[next_point] = True
                current = next_point
        
        return path
    
    def two_opt_swap(self, path: List[int], i: int, j: int) -> List[int]:
        """2-opt reversal operation"""
        new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
        return new_path
    
    def two_opt_optimize(self, waypoints: List[Coordinate3D], path: List[int]) -> List[int]:
        """2-opt local search optimization"""
        if not self.config.enable_local_search or len(path) < 4:
            return path
        
        improved = True
        best_distance = self.calculate_path_distance(self.extract_path(waypoints, path))
        current_path = path.copy()
        n = len(current_path)
        
        while improved:
            improved = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    # Try swapping
                    new_path = self.two_opt_swap(current_path, i, j)
                    new_distance = self.calculate_path_distance(self.extract_path(waypoints, new_path))
                    
                    if new_distance < best_distance:
                        best_distance = new_distance
                        current_path = new_path
                        improved = True
                        break
                if improved:
                    break
        
        return current_path
    
    def extract_path(self, waypoints: List[Coordinate3D], indices: List[int]) -> List[Coordinate3D]:
        """Extract coordinate path based on indices"""
        return [waypoints[idx] for idx in indices]
    
    def simulated_annealing(self, waypoints: List[Coordinate3D]) -> List[int]:
        """Simulated annealing algorithm to solve path"""
        n = len(waypoints)
        if n <= 1:
            return list(range(n))
        
        # Generate initial solution using greedy algorithm
        current_path = self.construct_greedy_path(waypoints)
        best_path = current_path.copy()
        current_dist = self.calculate_path_distance(self.extract_path(waypoints, current_path))
        best_dist = current_dist
        
        temp = self.config.initial_temperature
        cooling_rate = self.config.cooling_rate
        max_iter = min(self.config.max_iterations, n * n * 50)
        
        for iter in range(max_iter):
            # Randomly select two different points (excluding start)
            i = random.randint(1, n - 1)
            j = random.randint(1, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            # Try swapping
            new_path = self.two_opt_swap(current_path, i, j)
            new_dist = self.calculate_path_distance(self.extract_path(waypoints, new_path))
            
            # Calculate acceptance probability
            delta = new_dist - current_dist
            if delta < 0 or (temp > 1e-9 and random.random() < math.exp(-delta / temp)):
                current_path = new_path
                current_dist = new_dist
                if new_dist < best_dist:
                    best_dist = new_dist
                    best_path = current_path.copy()
            
            temp *= cooling_rate
            if temp < 1.0:
                break
        
        return best_path
    
    def solve_exact_internal(self, waypoints: List[Coordinate3D]) -> List[int]:
        """Exact solution method (only for small-scale problems)"""
        n = len(waypoints)
        if n > 12:
            return self.simulated_annealing(waypoints)
        
        # Use itertools to generate all permutations
        from itertools import permutations
        
        indices = list(range(n))
        best_perm = indices.copy()
        best_dist = self.calculate_path_distance(self.extract_path(waypoints, indices))
        
        # Enumerate all permutations to find optimal solution
        for perm in permutations(indices):
            perm_list = list(perm)
            dist = self.calculate_path_distance(self.extract_path(waypoints, perm_list))
            if dist < best_dist:
                best_dist = dist
                best_perm = perm_list
        
        return best_perm
    
    def plan_route(self, starts: List[Coordinate3D], 
                   waypoints: List[Coordinate3D], 
                   ends: List[Coordinate3D]) -> RouteResult:
        """Main route planning function"""
        start_time = time.time()
        
        best_full_path: List[Coordinate3D] = []
        best_total_distance = float('inf')
        best_start_idx = 0
        best_end_idx = 0
        
        # Iterate all possible start and end point combinations
        for si in range(len(starts)):
            for ei in range(len(ends)):
                # Solve middle path
                if len(waypoints) <= 12:
                    internal_order = self.solve_exact_internal(waypoints)
                else:
                    internal_order = self.simulated_annealing(waypoints)
                    if self.config.enable_local_search:
                        internal_order = self.two_opt_optimize(waypoints, internal_order)
                
                # Build complete path
                full_path = [starts[si]]
                for idx in internal_order:
                    full_path.append(waypoints[idx])
                full_path.append(ends[ei])
                
                # Update optimal path
                total_dist = self.calculate_path_distance(full_path)
                if total_dist < best_total_distance:
                    best_total_distance = total_dist
                    best_full_path = full_path
                    best_start_idx = si
                    best_end_idx = ei
        
        end_time = time.time()
        exec_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        return RouteResult(
            path=best_full_path,
            total_distance=best_total_distance,
            execution_time_ms=exec_time,
            start_index=best_start_idx,
            end_index=best_end_idx
        )
    
    def clear_distance_cache(self):
        self.dist_calc.clear_distance_cache()

# ===== 3D Coordinate Generation and Import Functions =====
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
    """Import 3D coordinates from a single file with section markers
    
    The file should use comment markers to separate different types of points:
    - # START_POINTS: for start points section
    - # WAYPOINTS: for waypoints section  
    - # END_POINTS: for end points section
    
    Coordinates should be in x, y, z format separated by comma, space, or tab.
    
    Args:
        file_path: Path to the file containing coordinates
        
    Returns:
        Tuple of (start_points, waypoints, end_points) as lists of Coordinate3D objects
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file format is incorrect
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    start_points = []
    waypoints = []
    end_points = []
    
    # Default to start points section if no section markers found
    current_section = "START_POINTS"
    
    with open(file_path, 'r') as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            
            # Check for section markers
            if line.upper() == "# START_POINTS":
                current_section = "START_POINTS"
                continue
            elif line.upper() == "# WAYPOINTS":
                current_section = "WAYPOINTS"
                continue
            elif line.upper() == "# END_POINTS":
                current_section = "END_POINTS"
                continue
            
            # Skip other comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Try different delimiters (comma, space, tab)
            valid_coordinate = False
            for delimiter in [',', ' ', '\t']:
                parts = line.split(delimiter)
                # Remove empty strings if any
                parts = [p.strip() for p in parts if p.strip()]
                
                if len(parts) >= 3:
                    try:
                        # Extract x, y, z coordinates
                        x = float(parts[0])
                        y = float(parts[1])
                        z = float(parts[2])
                        coord = Coordinate3D(x, y, z)
                        
                        # Add to the appropriate section
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
                raise ValueError(f"Invalid coordinate format at line {line_idx}: {line}")
    
    # Validate that we have at least one start and one end point
    if not start_points:
        start_points = [Coordinate3D(0, 0, 0)]  # Default start point
        print("Warning: No start points found, using default (0, 0, 0)")
    
    if not end_points:
        end_points = [Coordinate3D(100, 100, 50)]  # Default end point
        print("Warning: No end points found, using default (100, 100, 50)")
    
    return start_points, waypoints, end_points

def print_coordinates_3d(coords: List[Coordinate3D], title: str):
    """Print 3D coordinates"""
    print(f"===== {title} =====")
    for i, coord in enumerate(coords):
        print(f"  Point {i+1}: X={coord.x:.6f}, Y={coord.y:.6f}, Z={coord.z:.6f}")

def print_result_3d(result: RouteResult):
    """Print 3D route planning result"""
    print("\n===== 3D Route Planning Result =====")
    print(f"Total Distance: {result.total_distance:.2f} units")
    print(f"Computation Time: {result.execution_time_ms:.0f} milliseconds")
    print(f"Route Sequence ({len(result.path)} points):")
    
    for i, point in enumerate(result.path):
        print(f"  {i+1}. X={point.x:.6f}, Y={point.y:.6f}, Z={point.z:.6f}")

def plot_route_3d(result: RouteResult, starts: List[Coordinate3D], 
                 waypoints: List[Coordinate3D], ends: List[Coordinate3D]):
    """Plot 3D route planning result using matplotlib
    
    Args:
        result: The route planning result
        starts: List of start points
        waypoints: List of waypoints
        ends: List of end points
    """
    try:
        # Create figure and 3D axis
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract path coordinates
        path_x = [p.x for p in result.path]
        path_y = [p.y for p in result.path]
        path_z = [p.z for p in result.path]
        
        # Plot start points (blue circles)
        start_x = [p.x for p in starts]
        start_y = [p.y for p in starts]
        start_z = [p.z for p in starts]
        ax.scatter(start_x, start_y, start_z, c='blue', s=100, marker='o', label='Start Points')
        
        # Plot waypoints (green squares)
        waypoint_x = [p.x for p in waypoints]
        waypoint_y = [p.y for p in waypoints]
        waypoint_z = [p.z for p in waypoints]
        ax.scatter(waypoint_x, waypoint_y, waypoint_z, c='green', s=80, marker='s', label='Waypoints')
        
        # Plot end points (red triangles)
        end_x = [p.x for p in ends]
        end_y = [p.y for p in ends]
        end_z = [p.z for p in ends]
        ax.scatter(end_x, end_y, end_z, c='red', s=90, marker='^', label='End Points')
        
        # Plot the optimal path (black line with arrows)
        ax.plot(path_x, path_y, path_z, 'k-', linewidth=2, alpha=0.7, label='Optimal Path')
        
        # Add arrows to indicate direction
        for i in range(len(path_x) - 1):
            ax.quiver(
                path_x[i], path_y[i], path_z[i],
                path_x[i+1] - path_x[i], path_y[i+1] - path_y[i], path_z[i+1] - path_z[i],
                length=0.8, color='black', arrow_length_ratio=0.3
            )
        
        # Mark selected start and end points
        ax.scatter([result.path[0].x], [result.path[0].y], [result.path[0].z], 
                  c='cyan', s=150, marker='o', edgecolors='black', linewidths=2, label='Selected Start')
        ax.scatter([result.path[-1].x], [result.path[-1].y], [result.path[-1].z], 
                  c='magenta', s=150, marker='^', edgecolors='black', linewidths=2, label='Selected End')
        
        # Add labels
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        ax.set_title('3D Route Planning Visualization')
        
        # Add legend
        ax.legend()
        
        # Add grid for better visualization
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set equal scaling
        max_range = max(
            max(path_x) - min(path_x),
            max(path_y) - min(path_y),
            max(path_z) - min(path_z)
        )
        mid_x = (max(path_x) + min(path_x)) / 2
        mid_y = (max(path_y) + min(path_y)) / 2
        mid_z = (max(path_z) + min(path_z)) / 2
        
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        # Add distance text
        plt.figtext(0.5, 0.01, f'Total Distance: {result.total_distance:.2f} units', 
                   ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        
        # Show plot
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error plotting 3D visualization: {e}")
        print("Please make sure matplotlib is installed correctly.")

# ===== Main Function =====
def main():
    # Configure parameters
    config = AlgorithmConfig(
        cooling_rate=0.995,
        initial_temperature=10000.0,
        max_iterations=100000,
        use_cache=True,
        enable_local_search=True
    )
    
    # Data input method selection
    print("Select data input method:")
    print("1. Generate random coordinates")
    print("2. Import coordinates from files")
    
    try:
        input_method = input("Enter your choice (1 or 2): ").strip()
        
        if input_method == '1':
            # Generate random coordinates
            num_starts = int(input("Enter number of start points: "))
            num_waypoints = int(input("Enter number of waypoints: "))
            num_ends = int(input("Enter number of end points: "))
            
            if num_starts <= 0 or num_waypoints < 0 or num_ends <= 0:
                print("Input numbers must be positive integers!")
                return
            
            # Generate random 3D coordinates (example range)
            starts = generate_random_coordinates_3d(num_starts, 0, 100, 0, 100, 0, 50)
            waypoints = generate_random_coordinates_3d(num_waypoints, 0, 100, 0, 100, 0, 50)
            ends = generate_random_coordinates_3d(num_ends, 0, 100, 0, 100, 0, 50)
            
        elif input_method == '2':
            # Import coordinates from a single file with section markers
            print("\nNote: The file should contain sections separated by comment markers:")
            print("      # START_POINTS - for start points")
            print("      # WAYPOINTS - for waypoints")
            print("      # END_POINTS - for end points")
            print("      Each coordinate line should contain x, y, z separated by comma, space, or tab")
            
            print("\n--- Import All Coordinates ---")
            coordinate_file = input("Enter path to coordinate file: ").strip()
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
                waypoints = []
                ends = [Coordinate3D(100, 100, 50)]
            
        else:
            print("Invalid choice! Please enter 1 or 2.")
            return
            
    except ValueError:
        print("Please enter valid integers!")
        return
    
    # Print all point information
    print("\n" + "="*50)
    print_coordinates_3d(starts, "Candidate Start Points")
    print_coordinates_3d(waypoints, "Waypoints")
    print_coordinates_3d(ends, "Candidate End Points")
    
    # Execute route planning
    planner = RoutePlanner3D(config)
    result = planner.plan_route(starts, waypoints, ends)
    print_result_3d(result)
    
    # Plot the 3D route visualization
    print("\nGenerating 3D visualization...")
    plot_route_3d(result, starts, waypoints, ends)
    
    # Clear cache
    planner.clear_distance_cache()

if __name__ == "__main__":
    main()