import matplotlib

matplotlib.use('Agg')
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import torch

def plot_point_cloud(point_clouds):
    # Takes a list of point cloud tensors and plots them
    if not isinstance(point_clouds, list):
        point_clouds = [point_clouds]

    colors = ['red', 'blue', 'green', 'yellow', 'orange']  # List of colors for each point cloud
    traces = []  # List to store individual traces for each point cloud

    for i, point_cloud in enumerate(point_clouds):
        if isinstance(point_cloud, np.ndarray):
            pass
        elif isinstance(point_cloud, torch.Tensor):
            point_cloud = point_cloud.numpy()

        x_data = point_cloud[:, 0]
        y_data = point_cloud[:, 1]
        z_data = point_cloud[:, 2]

        # Create a trace for each point cloud with a different color
        trace = go.Scatter3d(
            x=x_data,
            y=y_data,
            z=z_data,
            mode='markers',
            marker=dict(
                size=5,
                opacity=0.8,
                color=colors[i % len(colors)]  # Assign color based on the index of the point cloud
            ),
            name=f"Point Cloud {i + 1}"
        )
        traces.append(trace)

    # Create the layout
    layout = go.Layout(
        scene=dict(
            aspectmode='data'
        )
    )

    # Create the figure and add the traces to it
    fig = go.Figure(data=traces, layout=layout)

    # Show the figure
    fig.show()

def plot_3d_graph(graph, node_positions, atom_types=None, edge_size=2):

    # plot nodes
    x_data = node_positions[:, 0]
    y_data = node_positions[:, 1]
    z_data = node_positions[:, 2]

    color_palette = ['blue', 'red', 'green', 'orange']
    if atom_types is None:
        atom_types = [0] * len(node_positions)
    colors = [color_palette[i] for i in atom_types]

    node_trace = go.Scatter3d(
        x=x_data,
        y=y_data,
        z=z_data,
        mode='markers',
        marker=dict(
            size=5,
            opacity=0.5,
            color=colors
        ),
        name="nodes",
        customdata=np.arange(len(graph)).reshape(-1,1),
        hovertemplate='node_id:%{customdata[0]:.2f}',
    )

    # plot edges
    x_edges, y_edges, z_edges = [], [], []
    for i,j in graph.edges:
        x_edges.extend(
            [node_positions[i,0], node_positions[j,0], None]
        )
        y_edges.extend(
            [node_positions[i,1], node_positions[j,1], None]
        )
        z_edges.extend(
            [node_positions[i,2], node_positions[j,2], None]
        )

    edge_trace = go.Scatter3d(
        x=x_edges,
        y=y_edges,
        z=z_edges,
        mode='lines',
        line=dict(
            width=edge_size
        ),
        name="edges"
    )

    # Create the layout
    layout = go.Layout(
        scene=dict(
            aspectmode='data'
        )
    )

    # Create the figure and add the traces to it
    fig = go.Figure(data=[node_trace, edge_trace], layout=layout)

    # Show the figure
    fig.show()


