"""
Author: Amin Nouri
Supervisor: Javad Mohammad Zadeh

Project Description:
--------------------
This FastAPI application visualizes molecular graphs from the MUTAG dataset
using PyTorch Geometric, NetworkX, and PyVis. It calculates both local
(node-level) and global (graph-level) metrics for each molecule. Users
can interactively explore molecules, view metrics, and download data.

Features:
---------
1. Node-level metrics:
   - Degree, Betweenness, Closeness, Eigenvector, PageRank, Clustering,
     K-core, Triangles, Average Neighbor Degree, Harmonic, Eccentricity
2. Graph-level metrics:
   - Number of nodes, Number of edges, Density, Average degree, Average clustering,
     Diameter, Radius, Connected components
3. Interactive PyVis graph visualization with atom coloring
4. Downloadable GraphML and JSON node features for each molecule
"""

import os
import json
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
import networkx as nx
from pyvis.network import Network
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ----------------------------
# Setup and dataset load
# ----------------------------
"""
Load the MUTAG dataset using PyTorch Geometric TUDataset.
MUTAG is a benchmark molecular dataset used for graph classification.
"""
dataset = TUDataset(root='data/MUTAG', name='MUTAG')
print(f"Number of graphs in MUTAG: {len(dataset)}")

# Atom mapping for visualization
atom_names = ["C", "O", "N", "Cl", "F", "S", "Br"]
atom_colors = {
    "C": "lightblue", "O": "orange", "N": "lightgreen",
    "Cl": "yellow", "F": "pink", "S": "violet", "Br": "brown"
}

# ----------------------------
# FastAPI App Setup
# ----------------------------
"""
Initialize FastAPI and mount static folder for storing
graph HTML, GraphML, and JSON files.
"""
app = FastAPI()
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ----------------------------
# Utility functions
# ----------------------------
def safe_centrality(func, graph, **kwargs):
    """
    Safe wrapper to compute centrality metrics.
    If calculation fails (e.g., convergence issue in eigenvector),
    it returns an empty dictionary instead of crashing.
    """
    try:
        return func(graph, **kwargs)
    except Exception:
        return {}

def export_graph_files(nx_graph: nx.Graph, mol_index: int, atom_labels):
    """
    Export NetworkX graph to:
    1. GraphML format (graph structure)
    2. JSON (node features and atom types)
    """
    graphml_path = f"static/molecule_{mol_index}.graphml"
    json_path = f"static/molecule_{mol_index}_nodes.json"

    # Save GraphML, fallback if some data types not supported
    try:
        nx.write_graphml(nx_graph, graphml_path)
    except Exception:
        gcopy = nx.Graph()
        gcopy.add_nodes_from(nx_graph.nodes())
        gcopy.add_edges_from(nx_graph.edges())
        nx.write_graphml(gcopy, graphml_path)

    # Save node features and atom types
    nodes_info = []
    for n, data in nx_graph.nodes(data=True):
        nodes_info.append({
            "node": n,
            "atom": atom_labels.get(n, "?"),
            "features": data.get("x")
        })
    with open(json_path, "w") as f:
        json.dump(nodes_info, f, indent=2)

    return graphml_path, json_path

# ----------------------------
# Main page route
# ----------------------------
@app.get("/", response_class=HTMLResponse)
async def main(request: Request, mol_index: int = 0):
    """
    Main interactive page for molecule visualization.
    Users can:
    - View molecule graph
    - Navigate with next/previous molecule
    - Download GraphML and JSON node features
    """
    # Ensure mol_index is within bounds
    mol_index = max(0, min(len(dataset) - 1, mol_index))

    # Load PyG graph and convert to NetworkX
    pyg_graph = dataset[mol_index]
    nx_graph = to_networkx(pyg_graph, to_undirected=True, node_attrs=['x'])

    # Molecule label (Mutagenic / Non-Mutagenic)
    try:
        label = int(pyg_graph.y.item())
    except Exception:
        label = 0
    label_text = "Mutagenic" if label == 1 else "Non-Mutagenic"
    label_color = "#d9534f" if label == 1 else "#5cb85c"

    # Map node features to atoms
    atom_labels = {}
    for n, data in nx_graph.nodes(data=True):
        feature = data.get("x")
        atom_index = -1
        if isinstance(feature, (list, tuple)):
            if 1 in feature:
                atom_index = feature.index(1)
        else:
            try:
                arr = feature.tolist()
                if 1 in arr:
                    atom_index = arr.index(1)
            except Exception:
                atom_index = -1
        atom_labels[n] = atom_names[atom_index] if 0 <= atom_index < len(atom_names) else "?"

    # ----------------------------
    # Local Metrics per Node
    # ----------------------------
    """
    Compute all local node-level metrics using NetworkX:
    - Degree Centrality
    - Betweenness Centrality
    - Closeness Centrality
    - Eigenvector Centrality
    - PageRank
    - Clustering Coefficient
    - K-core Number
    - Number of Triangles
    - Average Neighbor Degree
    - Harmonic Centrality
    - Eccentricity
    """
    degree_centrality = safe_centrality(nx.degree_centrality, nx_graph)
    betweenness = safe_centrality(nx.betweenness_centrality, nx_graph)
    closeness = safe_centrality(nx.closeness_centrality, nx_graph)
    eigenvector = safe_centrality(nx.eigenvector_centrality, nx_graph, max_iter=500)
    pagerank = safe_centrality(nx.pagerank, nx_graph)
    clustering = safe_centrality(nx.clustering, nx_graph)
    kcore = safe_centrality(nx.core_number, nx_graph)
    triangles = safe_centrality(nx.triangles, nx_graph)
    avg_neighbor_degree = safe_centrality(nx.average_neighbor_degree, nx_graph)
    harmonic = safe_centrality(nx.harmonic_centrality, nx_graph)
    eccentricity = safe_centrality(nx.eccentricity, nx_graph)

    # ----------------------------
    # Global Graph Metrics
    # ----------------------------
    """
    Compute graph-level metrics:
    - Number of nodes
    - Number of edges
    - Density
    - Average degree
    - Average clustering
    - Diameter and radius (if connected)
    - Connected components
    """
    global_metrics = {
        'Number of nodes': nx.number_of_nodes(nx_graph),
        'Number of edges': nx.number_of_edges(nx_graph),
        'Graph density': nx.density(nx_graph),
        'Average degree': round(sum(dict(nx_graph.degree()).values()) / max(1, nx.number_of_nodes(nx_graph)), 4),
        'Average clustering': round(nx.average_clustering(nx_graph), 4),
        'Diameter': nx.diameter(nx_graph) if nx.is_connected(nx_graph) else "N/A",
        'Radius': nx.radius(nx_graph) if nx.is_connected(nx_graph) else "N/A",
        'Connected components': nx.number_connected_components(nx_graph)
    }

    # ----------------------------
    # PyVis Interactive Graph
    # ----------------------------
    """
    Create an interactive PyVis network visualization:
    - Node color represents atom type
    - Physics simulation enabled
    - Save as HTML for iframe embedding
    """
    interactive_graph = Network(height="600px", width="100%", notebook=False, bgcolor="#ffffff")
    interactive_graph.toggle_physics(True)
    for n, data in nx_graph.nodes(data=True):
        atom_label = atom_labels.get(n, "?")
        color = atom_colors.get(atom_label, "lightgray")
        interactive_graph.add_node(n, label=str(n), title=f"Atom: {atom_label}", color=color)
    for source, target in nx_graph.edges():
        interactive_graph.add_edge(source, target)
    pyvis_html = f"static/molecule_{mol_index}.html"
    interactive_graph.write_html(pyvis_html)

    # Export GraphML & JSON
    graphml_path, json_path = export_graph_files(nx_graph, mol_index, atom_labels)
    graphml_link = f"/{graphml_path}"
    json_link = f"/{json_path}"
    html_graph_url = f"/{pyvis_html}"

    # Previous / Next molecule indices
    prev_idx = max(0, mol_index - 1)
    next_idx = min(len(dataset) - 1, mol_index + 1)

    # ----------------------------
    # Build HTML Page
    # ----------------------------
    """
    Full HTML page:
    - Left panel: interactive graph
    - Right panel: global metrics table
    - Below: local metrics table (node-level)
    - Controls for molecule index input and navigation
    - Download links for GraphML and JSON
    """
    full_html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>MUTAG Molecule Viewer</title>
<style>
body {{ font-family: Arial, Helvetica, sans-serif; margin:16px; background:#f7f9fb; color:#222; }}
.header {{ text-align:center; margin-bottom:12px; }}
.author, .supervisor {{ font-size:13px; color:#555; margin-top:4px; }}
.badge {{ display:inline-block; padding:8px 16px; border-radius:12px; color:white; font-weight:600; box-shadow:0 2px 6px rgba(0,0,0,0.12); }}
.controls {{ text-align:center; margin-bottom:16px; }}
.controls input[type=number] {{ width:90px; padding:6px; margin-right:6px; }}
.controls button {{ padding:6px 10px; margin-left:6px; }}
.container {{ display:flex; gap:18px; align-items:flex-start; }}
.left {{ flex:0 0 65%; box-shadow:0 6px 18px rgba(0,0,0,0.08); background:white; padding:8px; border-radius:6px; }}
.right {{ flex:0 0 35%; box-shadow:0 6px 18px rgba(0,0,0,0.06); background:white; padding:12px; border-radius:6px; }}
iframe {{ width:90%; height:520px; border:none; border-radius:4px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th, td {{ padding:8px; border:1px solid #e7ecf0; text-align:center; }}
th {{ background:#f0f4f7; }}
.metrics-title {{ font-size:18px; margin-bottom:8px; font-weight:600; }}
.local-section {{ margin-top:18px; background:white; padding:12px; border-radius:6px; box-shadow:0 6px 18px rgba(0,0,0,0.04); }}
.download-links a {{ display:inline-block; margin-right:10px; padding:6px 10px; background:#3b82f6; color:white; text-decoration:none; border-radius:6px; }}
.small-muted {{ font-size:12px; color:#666; }}
</style>
</head>
<body>
<div class="header" style="text-align:center;">
  <h1 style="font-size:24px;">MUTAG Molecule Viewer</h1>
  <div style="background:#3b82f6; color:white; padding:10px 14px; border-radius:10px; line-height:1.4; display:inline-block; margin-top:8px;">
    <h3 style="margin:0; font-size:16px;">Academic Supervisor: Dr. Javad Mohammad Zadeh</h3>
    <h3 style="margin:0; font-size:16px;">Implementation by Amin Nouri</h3>
  </div>
  <div style="height:6px;"></div>
  <div style="background:{label_color}; color:white; padding:6px 12px; border-radius:8px; font-size:14px; display:block; width:fit-content; margin:0 auto; text-align:center;">
    {label_text}
  </div>
</div>




<div class="controls">
<form method="get" style="display:inline-block;">
<label class="small-muted">Enter molecule index (0-{len(dataset)-1}):</label>
<input type="number" name="mol_index" min="0" max="{len(dataset)-1}" value="{mol_index}">
<input type="submit" value="View">
</form>
<a href="/?mol_index={prev_idx}"><button>&laquo; Previous</button></a>
<a href="/?mol_index={next_idx}"><button>Next &raquo;</button></a>
<span style="margin-left:12px;" class="download-links">
<a href="{graphml_link}" download>Download GraphML</a>
<a href="{json_link}" download>Download Nodes JSON</a>
</span>
</div>

<div class="container">
<div class="left">
<iframe src="{html_graph_url}"></iframe>
</div>
<div class="right">
<div class="metrics-title">Global Metrics</div>
<table>"""
    for k, v in global_metrics.items():
        full_html += f"<tr><td style='text-align:left;font-weight:600'>{k}</td><td>{v}</td></tr>"
    full_html += f"""</table>
<div style="margin-top:12px;" class="small-muted">
Node count: {global_metrics['Number of nodes']} &middot; Edge count: {global_metrics['Number of edges']}
</div>
</div>
</div>

<div class="local-section">
<div style="font-weight:700;margin-bottom:8px;">Local Metrics (per Node)</div>
<table>
<tr>
<th>Node</th><th>Atom</th><th>Degree</th><th>Betweenness</th><th>Closeness</th>
<th>Eigenvector</th><th>PageRank</th><th>Clustering</th><th>K-core</th>
<th>Triangles</th><th>AvgNbrDeg</th><th>Harmonic</th><th>Eccentricity</th>
</tr>"""
    for n, data in nx_graph.nodes(data=True):
        atom_label = atom_labels.get(n, "?")
        full_html += f"""
<tr>
<td>{n}</td>
<td>{atom_label}</td>
<td>{degree_centrality.get(n,0):.4f}</td>
<td>{betweenness.get(n,0):.4f}</td>
<td>{closeness.get(n,0):.4f}</td>
<td>{eigenvector.get(n,0):.4f}</td>
<td>{pagerank.get(n,0):.4f}</td>
<td>{clustering.get(n,0):.4f}</td>
<td>{kcore.get(n,0)}</td>
<td>{triangles.get(n,0)}</td>
<td>{avg_neighbor_degree.get(n,0):.4f}</td>
<td>{harmonic.get(n,0):.4f}</td>
<td>{eccentricity.get(n,0)}</td>
</tr>"""
    full_html += """
</table>
</div>

<div style="margin-top:18px; font-size:12px; color:#666;">
<strong>Note:</strong> Atom assignment is inferred from the node feature vector present in the dataset.
</div>
</body>
</html>"""

    return HTMLResponse(content=full_html)

# ----------------------------
# Download Endpoints
# ----------------------------
@app.get("/download/graphml/{mol_index}")
async def download_graphml(mol_index: int):
    """
    Serve the GraphML file of the selected molecule for download.
    """
    path = f"static/molecule_{mol_index}.graphml"
    if os.path.exists(path):
        return FileResponse(path, media_type="application/graphml+xml", filename=os.path.basename(path))
    return HTMLResponse(content=f"<h3>Not found: {path}</h3>", status_code=404)

@app.get("/download/nodes_json/{mol_index}")
async def download_nodes_json(mol_index: int):
    """
    Serve the node features JSON file of the selected molecule for download.
    """
    path = f"static/molecule_{mol_index}_nodes.json"
    if os.path.exists(path):
        return FileResponse(path, media_type="application/json", filename=os.path.basename(path))
    return HTMLResponse(content=f"<h3>Not found: {path}</h3>", status_code=404)
