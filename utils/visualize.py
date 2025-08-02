import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import List


def plot_eigenvalues(eigenvalues: List[float], title: str = "Eigenvalue Spectrum") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(1, len(eigenvalues) + 1)),
        y=eigenvalues,
        marker_color=[
            f"rgba(99, 110, 250, {max(0.3, v / max(eigenvalues))})"
            for v in eigenvalues
        ],
        name="Eigenvalues"
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Component",
        yaxis_title="Eigenvalue",
        template="plotly_dark",
        height=350,
    )
    return fig


def plot_spectral_probs(probs: List[float]) -> go.Figure:
    """Pie-like distribution of normalized eigenvalues."""
    labels = [f"λ{i+1}" for i in range(len(probs))]
    fig = go.Figure(go.Bar(
        x=labels,
        y=probs,
        marker_color=px.colors.sequential.Plasma_r[:len(probs)],
    ))
    fig.update_layout(
        title="SpectralEntropy — Eigenvalue Probability Distribution",
        xaxis_title="Eigenvalue",
        yaxis_title="Normalized Weight",
        template="plotly_dark",
        height=350,
    )
    return fig


