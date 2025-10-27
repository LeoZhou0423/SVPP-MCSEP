# SVPP-MCSEP: Simulated Vehicle Path Planning for Multi-Start Multi-End Route Problem

This program implements a 3D route planning algorithm that finds the optimal path connecting start points, waypoints, and end points using a hybrid approach combining greedy algorithms, simulated annealing, and 2-opt local search. The program now includes 3D visualization capabilities to visually display planning results.

## New Feature: Data Import Functionality

### Overview
The program now supports importing coordinate data from a single file, using special section markers to distinguish between different types of points. This makes it easier to reproduce results and work with predefined datasets.

### File Format
The file must be organized with the following section markers:

1. `# START_POINTS` - Indicates the start of start points section
2. `# WAYPOINTS` - Indicates the start of waypoints section  
3. `# END_POINTS` - Indicates the start of end points section

- Each coordinate line should contain x, y, z values separated by comma, space, or tab
- Empty lines and other comment lines (starting with `#`) are ignored

All three section markers must be present in the file. The program will throw an error if any section marker is missing.

### Example File Format
```
# START_POINTS
0, 0, 0       # Default start point
10, 15, 5     # Alternative start point

# WAYPOINTS
30, 30, 20    # Waypoint 1
40, 45, 25    # Waypoint 2

# END_POINTS
80, 70, 40    # End point 1
100, 100, 50  # Default end point
```

A sample file `example_coordinates.txt` is included in the repository.

## Usage

1. Run the program:
   ```
   python HOAMRP_Planner.py
   ```

2. Choose the data input method:
   - Enter `1` to generate random coordinates
     - Specify the number of start points, waypoints, and end points
     - Minimum values: 1 start point, 0 waypoints, 1 end point
     - Random coordinates are generated within ranges: x ∈ [0,100], y ∈ [0,100], z ∈ [0,50]
   - Enter `2` to import coordinates from files

3. If importing from a file:
   - Provide the file path for your coordinate file containing all point types
   - The file must use section markers (# START_POINTS, # WAYPOINTS, # END_POINTS) to separate point types
   - All three section markers are required
   - You can leave the file path empty to use default values:
     - Default start point: (0, 0, 0)
     - Default end point: (100, 100, 50)
     - No waypoints by default

4. The program will display the imported/generated coordinates and compute the optimal route

## Algorithm Details

- **Multiple algorithm options**: Exact solution for small problems (≤ 8 waypoints), simulated annealing with 2-opt local search for medium problems, and clustering + ALNS approach for large problems (≥ 50 waypoints)
- **Optimization techniques**: Vectorized distance calculations, KD-Tree spatial indexing, and distance caching for improved performance
- **Simulated annealing parameters**: Fixed parameters (T₀ = 5000, α = 0.998) as specified in the paper
- **Greedy Construction**: Builds an initial path by always selecting the nearest unvisited point
- **2-opt Local Search**: Further optimizes the path by reversing segments
- **Distance Caching**: Optimizes performance by caching calculated distances

## Requirements

- Python 3.6+
- Standard libraries: math, random, time, os, typing, itertools
- External libraries: numpy, matplotlib (for 3D visualization)

To install the required dependencies, run:
```
pip install numpy matplotlib
```

## Example Workflow for Reproducibility

### Random Data Generation
1. Select option 1 to generate random coordinates
2. Specify the number of start points, waypoints, and end points
3. Random coordinates will be generated within ranges: x ∈ [0,100], y ∈ [0,100], z ∈ [0,50]

### File Import Method
1. Save your coordinate data in a single file with all required section markers:
   ```
   # START_POINTS
   0, 0, 0
   
   # WAYPOINTS
   20, 20, 10
   40, 40, 20
   60, 60, 30
   
   # END_POINTS
   100, 100, 50
   ```

   Note: All three section markers (# START_POINTS, # WAYPOINTS, # END_POINTS) must be present in the file.

2. Run the program and select option 2

3. Enter the file path when prompted

4. The program will compute the optimal route and display the text results

5. A 3D visualization window will automatically open, showing:
   - All candidate start points (blue circles)
   - All waypoints (green squares)
   - All candidate end points (red triangles)
   - The optimal path with directional arrows
   - Highlighted selected start (cyan) and end (magenta) points

## 3D Visualization Features

The 3D visualization provides an intuitive representation of the route planning result:

- **Color-coded points**: Different colors and shapes distinguish between start points, waypoints, and end points
- **Directional arrows**: Show the order of traversal along the path
- **Highlighted selection**: Clearly identifies which start and end points were selected for the optimal route
- **Equal scaling**: Ensures proper 3D perspective with consistent scaling across all axes
- **Interactive viewing**: You can rotate, zoom, and pan the 3D plot for better inspection
- **Distance information**: Displays the total path distance on the plot

To close the visualization window, simply press any key or click the close button.

This approach ensures that you can easily reproduce results by using the same input files.

## Output

The program displays:
- All imported/generated coordinates
- The optimal route sequence
- Total distance of the optimal route
- Computation time in milliseconds
- The indices of the selected start and end points

## Notes

- For large datasets, the simulated annealing algorithm is used instead of the exact solution
- Local search can be disabled in the `AlgorithmConfig` if needed
- Distance caching is enabled by default to improve performance
