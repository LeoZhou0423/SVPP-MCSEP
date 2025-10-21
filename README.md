# SVPP-MCSEP: Single Vehicle Path Planning with Multiple Candidate Start/End Points

A hybrid optimization algorithm for 3D path planning that efficiently solves the complex problem of finding optimal routes with multiple candidate start and end points.

## 🚀 Features

- **Hybrid Optimization**: Combines exact algorithms, simulated annealing, and 2-opt local search
- **3D Spatial Planning**: Supports Euclidean distance calculations in three-dimensional space
- **Flexible Start/End**: Choose optimal paths from multiple candidate starting and ending points
- **Adaptive Strategy**: Automatically switches between exact and heuristic methods based on problem size
- **Performance Optimized**: Implements distance caching and efficient data structures

## 📦 Installation

```bash
git clone https://github.com/LeoZhou0423/svpp-mcsep.git
cd svpp-mcsep
pip install -r requirements.txt
