import numpy as np
import plotly.express as px
import threading
import secrets

import dash
from dash import dcc, html, Input, Output, State

from forest_ca import Forest

# -----------------------------------------------------------------------------
# Colourmap for the four cell states (empty, good, burning, burned)
# -----------------------------------------------------------------------------
COLORSCALE = [
    [0.00, "black"], [0.24, "black"],            # Empty
    [0.24, "forestgreen"], [0.49, "forestgreen"],# Good tree
    [0.49, "orangered"],   [0.74, "orangered"],  # Burning tree
    [0.74, "lightgray"],   [1.00, "lightgray"],  # Burned tree
]

# -----------------------------------------------------------------------------
# Initial parameters (keep them in one place)
# -----------------------------------------------------------------------------
DEFAULTS = dict(
    width          = 100,
    height         = 100,
    p_tree         = 0.4,
    wind_speed     = 2.0,
    wind_dir       = (0, 0),          # calm
    density        = 30,              # cluster density
    noise_octaves  = 30,
    seed           = secrets.randbelow(1_000_000),
)

# -----------------------------------------------------------------------------
# Global simulation object (will be re-created on reset)
# -----------------------------------------------------------------------------
forest_lock = threading.Lock()
forest = Forest(
    (DEFAULTS["width"], DEFAULTS["height"]),
    density       = DEFAULTS["density"],
    noise_octaves = DEFAULTS["noise_octaves"],
    p_tree        = DEFAULTS["p_tree"],
    wind_speed    = DEFAULTS["wind_speed"],
    wind_dir      = DEFAULTS["wind_dir"],
    seed          = DEFAULTS["seed"],
)

# -----------------------------------------------------------------------------
# Dash application
# -----------------------------------------------------------------------------
app = dash.Dash(__name__)
app.title = "Forest-Fire CA – Live"


def make_board_fig(board: np.ndarray):
    """Render the CA board as a Plotly figure."""
    fig = px.imshow(
        board,
        color_continuous_scale=COLORSCALE,
        zmin=0, zmax=3,
        origin="lower",
        aspect="equal",
    )
    fig.update_layout(
        margin=dict(l=5, r=5, t=5, b=5),
        coloraxis_showscale=False,
        xaxis_visible=False,
        yaxis_visible=False,
    )
    return fig

# -----------------------------------------------------------------------------
# Layout – left pane controls, right pane animation
# -----------------------------------------------------------------------------
app.layout = html.Div([
    # ─────────────────────────── Controls ────────────────────────────────
    html.Div([
        html.H2("Controls", className="font-bold text-lg"),

        # ▲ Board size ---------------------------------------------------
        html.Label("Board width"),
        dcc.Input(id="width", type="number", min=10, max=300, step=10,
                  value=DEFAULTS["width"], className="w-full"),
        html.Br(),
        html.Label("Board height"),
        dcc.Input(id="height", type="number", min=10, max=300, step=10,
                  value=DEFAULTS["height"], className="w-full"),

        html.Hr(),

        # ▲ Per-cell tree probability ------------------------------------
        html.Label("Tree probability (p_tree)"),
        dcc.Slider(id="p-tree", min=0.05, max=0.95, step=0.05,
                   value=DEFAULTS["p_tree"],
                   tooltip={"always_visible": True}),

        html.Br(),
        html.Label("Wind speed"),
        dcc.Slider(id="wind-speed", min=0.0, max=5.0, step=0.5,
                   value=DEFAULTS["wind_speed"],
                   tooltip={"always_visible": True}),

        html.Br(),
        html.Label("Wind direction"),
        dcc.Dropdown(
            id="wind-dir", clearable=False, value="0,0",
            options=[
                {"label": "↔︎ calm",   "value": "0,0"},
                {"label": "→  (E)",   "value": "0,1"},
                {"label": "↗︎ (NE)",  "value": "1,1"},
                {"label": "↑  (N)",   "value": "1,0"},
                {"label": "↖︎ (NW)",  "value": "1,-1"},
                {"label": "←  (W)",   "value": "0,-1"},
                {"label": "↙︎ (SW)",  "value": "-1,-1"},
                {"label": "↓  (S)",   "value": "-1,0"},
                {"label": "↘︎ (SE)",  "value": "-1,1"},
            ]),

        html.Hr(),

        # ▲ Cluster density & noise octaves ------------------------------
        html.Label("Cluster density"),
        dcc.Slider(id="cluster-density", min=1, max=100, step=1,
                   value=DEFAULTS["density"],
                   tooltip={"always_visible": True}),
        html.Br(),
        html.Label("Noise octaves"),
        dcc.Slider(id="noise-octaves", min=1, max=60, step=1,
                   value=DEFAULTS["noise_octaves"],
                   tooltip={"always_visible": True}),

        html.Hr(),

        # ▲ Seed ---------------------------------------------------------
        html.Label("RNG seed (leave blank ⇒ random)"),
        dcc.Input(id="seed", type="number", placeholder="auto",
                  className="w-full"),

        html.Br(), html.Br(),
        html.Button("Reset simulation", id="btn-reset", n_clicks=0,
                    className="px-4 py-2 bg-blue-500 text-white rounded"),
        html.Button("Pause / Play", id="btn-toggle", n_clicks=0,
                    className="ml-2 px-4 py-2 bg-gray-500 text-white rounded"),
    ], style={"width": "24%", "display": "inline-block",
              "verticalAlign": "top", "padding": "10px"}),

    # ───────────────────────────── Animation ─────────────────────────────
    html.Div([
        dcc.Graph(id="forest-graph", figure=make_board_fig(forest.board)),
        dcc.Interval(id="interval", interval=200, n_intervals=0),
        # Hidden div for triggering reset without raising exception
        html.Div(id="_dummy", style={"display": "none"}),
    ], style={"width": "74%", "display": "inline-block"}),
])


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------

# ➜ Animation – advance CA & update figure every interval tick
@app.callback(
    Output("forest-graph", "figure"),
    Input("interval", "n_intervals"),
)
def update_graph(_):
    with forest_lock:
        forest.step()
        board = forest.board.copy()
    return make_board_fig(board)


# ➜ Pause / resume when button clicked
@app.callback(
    Output("interval", "disabled"),
    Input("btn-toggle", "n_clicks"),
    State("interval", "disabled"),
    prevent_initial_call=True,
)
def toggle_running(_, disabled):
    # Each click flips the boolean
    return not disabled


# ➜ Reset simulation with current parameter values
@app.callback(
    Output("_dummy", "children"),          # dummy output
    Output("forest-graph", "figure", allow_duplicate=True),      # refresh immediately
    Input("btn-reset", "n_clicks"),
    # State values in the same order as the controls above
    State("width", "value"),
    State("height", "value"),
    State("p-tree", "value"),
    State("cluster-density", "value"),
    State("noise-octaves", "value"),
    State("wind-speed", "value"),
    State("wind-dir", "value"),
    State("seed", "value"),
    prevent_initial_call=True,
)
def reset_sim(_, width, height, p_tree, density, noise_octaves,
              wind_speed, wind_dir, seed_input):
    global forest

    # Parse wind_dir "x,y" → (x, y)
    wdx, wdy = map(float, wind_dir.split(","))

    # If user left seed blank, pick a fresh one
    seed = int(seed_input) if seed_input is not None else secrets.randbelow(1_000_000)

    with forest_lock:
        forest = Forest(
            (int(width), int(height)),
            density       = int(density),
            noise_octaves = int(noise_octaves),
            p_tree        = float(p_tree),
            wind_speed    = float(wind_speed),
            wind_dir      = (wdx, wdy),
            seed          = seed,
        )
        board = forest.board.copy()

    return f"reset @ seed {seed}", make_board_fig(board)


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
