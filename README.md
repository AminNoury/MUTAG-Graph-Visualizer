# MUTAG Molecule Viewer

**Interactive MUTAG molecule visualizer using FastAPI, PyTorch Geometric, NetworkX, and PyVis.**

## Overview

This project provides a web-based interactive visualization for the [MUTAG dataset](https://chrsmrrs.github.io/datasets/) of chemical graphs. Features include:

- **Node-level metrics**: degree centrality, betweenness, closeness, eigenvector, PageRank, clustering, k-core, triangles, harmonic centrality, eccentricity.
- **Graph-level metrics**: number of nodes/edges, density, average degree, average clustering, diameter, radius, connected components.
- **Interactive visualization**: rendered with PyVis and displayed in a browser.
- **Data export**: download GraphML and JSON node features.
- **Built with**: FastAPI, PyTorch Geometric, NetworkX, PyVis.

## Author & Supervisor

- **Implementation**: Amin Nouri  
- **Supervisor**: Javad Mohammad Zadeh

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/mutag-molecule-viewer.git
cd mutag-molecule-viewer

2. Create a virtual environment and activate it:

pip install -r requirements.txt

3. The MUTAG dataset will be downloaded automatically on the first run

Usage

Run the FastAPI application:

uvicorn fastapi_mutag_app:app --reload


.

Use the input box or Next / Previous buttons to navigate molecules.

Download GraphML and JSON node feature files directly from the interface.

Features

Node Metrics: centrality measures, clustering, k-core, triangles, harmonic centrality, eccentricity.

Graph Metrics: density, diameter, connected components, average degree, average clustering.

Interactive PyVis Graph: zoom, drag nodes, hover for atom info.

Data Export: GraphML and JSON for external analysis.

Extensible: easy to add new metrics or datasets.

License

MIT License