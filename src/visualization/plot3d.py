from typing import Tuple, Optional

import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def scatter3d(features: np.ndarray,
              title: str = "3D plot",
              labels: Optional[Tuple[str, str, str]] = None,
              colors: Optional[np.ndarray] = None,
              custom_labels: Optional[np.ndarray] = None
              ):
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    if labels is None:
        xaxis_title, yaxis_title, zaxis_title = 'X', 'Y', 'Z'
    else:
        xaxis_title, yaxis_title, zaxis_title = labels

    scatter = go.Scatter3d(
        x=features[:, 0],
        y=features[:, 1],
        z=features[:, 2],
        mode='markers',
        marker=dict(
            size=8,
            color=colors,  # Użyj kolorów z parametru colors
            opacity=0.8
        ),
        text=custom_labels  # Ustaw własne etykiety
    )

    fig.add_trace(scatter)

    fig.update_layout(scene=dict(aspectmode="cube"))

    # Dodaj opcję obrotu
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Rotate",
                         method="animate",
                         args=[None, dict(frame=dict(duration=1000, redraw=True), fromcurrent=True)]),
                    dict(label="Stop",
                         method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=True), mode="immediate",
                                            transition=dict(duration=0))])
                ]
            )
        ]
    )

    fig.update_layout(scene=dict(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        zaxis_title=zaxis_title
    ))

    fig.update_layout(title_text=title)

    fig.show()
