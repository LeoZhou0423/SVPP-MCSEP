# Fast 3D Path Planner with Multiple Candidate Start/End Points

## Overview

**FastRoutePlanner3D** is an optimized Python implementation for solving 3D path planning problems with multiple candidate start and end points. This hybrid algorithm combines simulated annealing, adaptive large neighborhood search (ALNS), spatial indexing, and intelligent clustering to efficiently compute near-optimal routes through 3D waypoints.

## Key Features

- **3D Path Optimization**: Handles X, Y, Z coordinates for true 3D route planning
- **Multiple Start/End Points**: Supports multiple candidate starting and ending positions
- **Hybrid Algorithm**: Combines simulated annealing, ALNS, and local search
- **Performance Optimized**: Uses spatial indexing, distance caching, and parallel processing
- **Smart Clustering**: Automatically clusters waypoints for large-scale problems
- **Visualization**: 3D plotting with matplotlib for result analysis
- **Flexible Input**: Generate random coordinates or import from file

## Algorithm Architecture

### Core Components

1. **FastDistanceCalculator3D**
   - Vectorized distance computations
   - Precomputed distance matrix caching
   - Batch distance queries for performance

2. **FastSpatialIndex3D**
   - KD-tree based spatial indexing
   - Efficient nearest neighbor searches
   - Radius-based point queries

3. **FastALNSOperators**
   - Greedy nearest neighbor construction
   - Random and worst-edge destroy operators
   - Greedy repair operations

4. **FastRoutePlanner3D**
   - Multi-stage optimization pipeline
   - Adaptive algorithm selection based on problem size
   - Parallel processing for multiple start-end combinations

### Optimization Techniques

- **Simulated Annealing**: Temperature-controlled search with 2-opt neighborhood
- **Adaptive Large Neighborhood Search**: Dynamic destroy/repair operations
- **Hierarchical Clustering**: Agglomerative clustering for large problems
- **Early Termination**: Stops optimization when no improvement detected
- **Multi-stage Processing**: Different optimization intensities per stage

## Installation Requirements

```bash
pip install numpy scipy scikit-learn matplotlib
