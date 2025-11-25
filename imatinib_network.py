from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
import plotly.graph_objects as go

# Imatinib SMILES and targets

smiles_imatinib = "CC1=NC=NC(=N1)N2CCN(CC2)CC3=CC=CC=C3C(=O)NC4=CC=C(C=C4)N"
targets = {
    "ABL1": "CCNCCO",
    "KIT": "CC(C)O",
    "PDGFRB": "C1CCOC1",
    "DDR1": "CCN",
    "CSF1R": "CNC"
}
binding_atoms = {
    "ABL1": [(0, 0), (3, 1), (7, 2)],
    "KIT": [(0, 0), (3, 1)],
    "PDGFRB": [(0, 1), (3, 0)],
    "DDR1": [(0, 0)],
    "CSF1R": [(0, 0), (3, 1)]
}

# Build atomic graph function

def build_atomic_graph(smiles, node_type="atom"):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    G = nx.Graph()
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        G.add_node(idx, element=atom.GetSymbol(), type=node_type)
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                   bond_type=str(bond.GetBondType()), style='white_solid')
    return G, mol

# Generate line coordinates

def line_coords(p0, p1, style='solid', segments=12):
    if style == 'solid':
        return [p0[0], p1[0], None], [p0[1], p1[1], None], [p0[2], p1[2], None]
    x, y, z = [], [], []
    for i in range(segments):
        t0 = i / segments
        t1 = (i + 0.2) / segments
        x += [p0[0]*(1-t0)+p1[0]*t0, p0[0]*(1-t1)+p1[0]*t1, None]
        y += [p0[1]*(1-t0)+p1[1]*t0, p0[1]*(1-t1)+p1[1]*t1, None]
        z += [p0[2]*(1-t0)+p1[2]*t0, p0[2]*(1-t1)+p1[2]*t1, None]
    return x, y, z

# Plot Imatinib + target fragment

def plot_drug_target_atomic(G_drug, mol_drug, target_name, target_smiles):
    # Build target graph
    G_target, mol_target = build_atomic_graph(target_smiles, node_type="target")

    # Get positions once
    conf_drug = mol_drug.GetConformer()
    pos_drug = {n: (conf_drug.GetAtomPosition(n).x,
                     conf_drug.GetAtomPosition(n).y,
                     conf_drug.GetAtomPosition(n).z) for n in G_drug.nodes()}

    conf_target = mol_target.GetConformer()
    offset_z = 10
    target_start_idx = max(G_drug.nodes()) + 1
    pos_target = {n + target_start_idx: (conf_target.GetAtomPosition(n).x,
                                         conf_target.GetAtomPosition(n).y,
                                         conf_target.GetAtomPosition(n).z + offset_z)
                  for n in G_target.nodes()}

    # Relabel target nodes and combine graphs
    G_target_relabel = nx.relabel_nodes(G_target, lambda x: x + target_start_idx)
    G_combined = nx.compose(G_drug, G_target_relabel)

    # Add red dotted edges for binding atoms
    for d_idx, t_idx in binding_atoms.get(target_name, []):
        G_combined.add_edge(d_idx, t_idx + target_start_idx, style='red_dotted')

    # Merge positions
    pos_combined = {**pos_drug, **pos_target}

    # Edge traces
    
    edge_traces = []
    for u, v, attrs in G_combined.edges(data=True):
        style = attrs.get('style', 'white_solid')
        x, y, z = line_coords(pos_combined[u], pos_combined[v], style='dotted' if style=='red_dotted' else 'solid')
        color = 'red' if style=='red_dotted' else 'white'
        edge_traces.append(go.Scatter3d(x=x, y=y, z=z, mode='lines',
                                        line=dict(color=color, width=4), hoverinfo='none'))

    # Node traces

    node_x, node_y, node_z, node_text, node_colors = [], [], [], [], []
    for n, data in G_combined.nodes(data=True):
        x, y, z = pos_combined[n]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_text.append(data['element'])
        node_colors.append('lightblue' if data['type']=='atom' else 'lightgreen')

    node_trace = go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='markers+text',
                              marker=dict(symbol='circle', size=12, color=node_colors),
                              text=node_text, textposition="top center", hoverinfo='text')

    # Plot

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        paper_bgcolor='black',
        plot_bgcolor='black',
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False), bgcolor='black', aspectmode='data'),
        width=900, height=700,
        title=dict(text=f"Imatinib + {target_name}", font=dict(color='white')),
        showlegend=False, margin=dict(l=0,r=0,b=0,t=40)
    )
    fig.show()

# Main execution

G_drug, mol_drug = build_atomic_graph(smiles_imatinib)

for target_name, target_smiles in targets.items():
    plot_drug_target_atomic(G_drug, mol_drug, target_name, target_smiles)
