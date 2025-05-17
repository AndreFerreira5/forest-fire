import numpy as np
import plotly.express as px
import plotly.graph_objects as go          # NEW – avoids the pandas dependency
import threading, secrets, math

import dash
from dash import dcc, html, Input, Output, State

from forest_ca import Forest

# ─────────────────────────────── Colour map ──────────────────────────────
COLORSCALE = [
    [0.00, "black"], [0.24, "black"],
    [0.24, "forestgreen"], [0.49, "forestgreen"],
    [0.49, "orangered"],   [0.74, "orangered"],
    [0.74, "lightgray"],   [1.00, "lightgray"],
]

# ───────────────────────────── Default params ────────────────────────────
DEFAULTS = dict(
    width      = 100,
    height     = 100,
    p_tree     = 0.4,
    p_grow     = 1e-3,         # NEW
    f_light    = 1e-5,         # NEW
    wind_speed = 2.0,
    wind_dir   = (0, 0),
    density    = 30,
    noise_oct  = 30,
    rad_decay  = 0.4,          # NEW
    ign_base   = 0.8,          # NEW
    max_dist   = 3,            # NEW
    spot_prob  = 0.01,         # NEW
    spot_range = 15,           # NEW
    seed       = secrets.randbelow(1_000_000),
)

# ─────────────────────────── Global simulation ───────────────────────────
forest_lock = threading.Lock()
forest = Forest(
    (DEFAULTS["width"], DEFAULTS["height"]),
    p_tree        = DEFAULTS["p_tree"],
    p_grow        = DEFAULTS["p_grow"],
    f_lightning   = DEFAULTS["f_light"],
    density       = DEFAULTS["density"],
    noise_octaves = DEFAULTS["noise_oct"],
    wind_speed    = DEFAULTS["wind_speed"],
    wind_dir      = DEFAULTS["wind_dir"],
    radiant_decay       = DEFAULTS["rad_decay"],
    ignition_base_prob  = DEFAULTS["ign_base"],
    max_ignition_distance = DEFAULTS["max_dist"],
    spotting_prob       = DEFAULTS["spot_prob"],
    spotting_range      = DEFAULTS["spot_range"],
    seed          = DEFAULTS["seed"],
)

# ───────────────────────────── Dash app ──────────────────────────────────
app = dash.Dash(__name__)
app.title = "Forest-Fire CA – Live metrics"

def make_board_fig(board: np.ndarray):
    fig = px.imshow(board,
        color_continuous_scale=COLORSCALE,
        zmin=0, zmax=3, origin="lower", aspect="equal")
    fig.update_layout(
        margin=dict(l=5, r=5, t=5, b=5),
        coloraxis_showscale=False,
        xaxis_visible=False, yaxis_visible=False,
    )
    return fig

# ───────────────────────────── Layout ────────────────────────────────────
app.layout = html.Div([
    # ------------- controls ------------------------------------------------
    html.Div([
        html.H3("Controls"),
        # board size
        html.Label("Board width"),  dcc.Input(id="width", type="number",
            min=10, max=400, step=10, value=DEFAULTS["width"]),
        html.Label("Board height"), dcc.Input(id="height", type="number",
            min=10, max=400, step=10, value=DEFAULTS["height"]),
        html.Hr(),
        html.Label("Tree probability p_tree"),
        dcc.Slider(id="p-tree", min=0.05, max=0.95, step=0.05,
                   value=DEFAULTS["p_tree"], tooltip={"always_visible":True}),
        html.Label("Regrowth probability p_grow"),              # NEW
        dcc.Input(id="p-grow", type="number", value=DEFAULTS["p_grow"],
                  min=1e-6, max=0.1, step=1e-4),
        html.Label("Lightning probability f_light"),            # NEW
        dcc.Input(id="f-light", type="number", value=DEFAULTS["f_light"],
                  min=1e-8, max=1e-3, step=1e-6),
        html.Hr(),

        html.Label("Wind speed"),
        dcc.Slider(id="wind-speed", min=0, max=5, step=0.5,
                   value=DEFAULTS["wind_speed"], tooltip={"always_visible":True}),
        html.Label("Wind direction"),
        dcc.Dropdown(id="wind-dir", clearable=False, value="0,0",
            options=[{"label": lbl, "value": val} for lbl, val in [
                ("↔ calm", "0,0"),
                ("→ (E)",  "0,1"),  ("↗ NE", "1,1"), ("↑ (N)",  "1,0"),
                ("↖ NW",   "1,-1"), ("← (W)", "0,-1"),("↙ SW","-1,-1"),
                ("↓ (S)", "-1,0"),  ("↘ SE","-1,1"),
            ]]),
        html.Hr(),
        html.Label("Cluster density"),     dcc.Slider(id="density", min=1, max=100,
                   step=1, value=DEFAULTS["density"], tooltip={"always_visible":True}),
        html.Label("Perlin noise octaves"),dcc.Slider(id="noise-oct", min=1, max=60,
                   step=1, value=DEFAULTS["noise_oct"], tooltip={"always_visible":True}),
        html.Hr(),
        # spread parameters --------------------------------------------------
        html.Label("Radiant decay"),                           # NEW
        dcc.Slider(id="rad-decay", min=0.1, max=1.0, step=0.05,
                   value=DEFAULTS["rad_decay"], tooltip={"always_visible":True}),
        html.Label("Ignition base probability"),               # NEW
        dcc.Slider(id="ign-base", min=0.1, max=1.0, step=0.05,
                   value=DEFAULTS["ign_base"], tooltip={"always_visible":True}),
        html.Label("Max ignition distance"),                   # NEW
        dcc.Input(id="max-dist", type="number",
                  min=1, max=10, step=1, value=DEFAULTS["max_dist"]),
        html.Hr(),
        html.Label("Spotting probability"),                    # NEW
        dcc.Input(id="spot-prob", type="number",
                  min=0, max=0.2, step=0.01, value=DEFAULTS["spot_prob"]),
        html.Label("Spotting range (cells)"),                  # NEW
        dcc.Input(id="spot-range", type="number",
                  min=1, max=50, step=1, value=DEFAULTS["spot_range"]),
        html.Hr(),
        # seed
        html.Label("RNG seed (leave blank ⇒ random)"),
        dcc.Input(id="seed", type="number", placeholder="auto"),
        html.Br(), html.Br(),
        html.Button("Reset", id="btn-reset", n_clicks=0,
                    className="px-4 py-2 bg-blue-500 text-white rounded"),
        html.Button("Pause / Play", id="btn-toggle", n_clicks=0,
                    className="ml-2 px-4 py-2 bg-gray-500 text-white rounded"),
    ], style={"width": "23%", "display": "inline-block",
              "verticalAlign": "top", "padding": "10px", "overflowY":"auto"}),

    # ------------- visualisations -----------------------------------------
    html.Div([
        dcc.Graph(id="forest-graph", figure=make_board_fig(forest.board)),
        html.Div([
            dcc.Graph(id="cluster-hist", style={"height": 220, "display":"inline-block", "width":"49%"}),
            dcc.Graph(id="fire-hist",    style={"height": 220, "display":"inline-block", "width":"49%"}),
        ]),
        dcc.Graph(id="tree-stats",   style={"height": 240}),
        dcc.Graph(id="metric-stats", style={"height": 240}),
        html.H4(id="equil-text", style={"textAlign":"center"}),            # NEW
        dcc.Interval(id="interval", interval=200, n_intervals=0),
        html.Div(id="_dummy", style={"display":"none"}),
    ], style={"width": "75%", "display": "inline-block"}),
])

# ─────────────────────────── Callbacks – step ────────────────────────────
@app.callback(
    Output("forest-graph", "figure"),
    Output("cluster-hist", "figure"),
    Output("fire-hist",    "figure"),
    Output("tree-stats",   "figure"),
    Output("metric-stats", "figure"),
    Output("equil-text",   "children"),
    Input("interval", "n_intervals"),
)
def update_every_tick(_):
    with forest_lock:
        forest.step()
        board = forest.board.copy()

    # -------- cluster-size histogram (linear bins) ----------------------
    sizes, _ = forest.cluster_stats()
    if len(sizes):
        fig_cluster = go.Figure(go.Histogram(x=sizes, nbinsx=25))
    else:
        fig_cluster = go.Figure(go.Histogram(x=[0]))
    fig_cluster.update_layout(title="Cluster size distribution",
                              xaxis_title="size", yaxis_title="count",
                              margin=dict(l=20,r=20,b=30,t=30))

    # -------- cumulative fire-size histogram (log–log) -----------------
    if forest.fire_sizes:
        counts, edges = np.histogram(forest.fire_sizes, bins="auto")
        cdf = np.cumsum(counts[::-1])[::-1]
        fig_fire = go.Figure(go.Scatter(x=edges[:-1], y=cdf,
                                        mode="lines+markers"))
        fig_fire.update_layout(title="Cumulative fire-size (log–log)",
            xaxis=dict(title="size", type="log"),
            yaxis=dict(title="P(size ≥ s)", type="log"),
            margin=dict(l=20,r=20,b=30,t=30))
    else:
        fig_fire = go.Figure(go.Scatter())
        fig_fire.update_layout(title="No fires yet")

    # -------- tree-count time series -----------------------------------
    steps = np.arange(1, len(forest.alive_trees_hist)+1)
    fig_tree = go.Figure()
    fig_tree.add_trace(go.Scatter(x=steps, y=forest.alive_trees_hist,
                                  name="Alive"))
    fig_tree.add_trace(go.Scatter(x=steps, y=forest.burning_trees_hist,
                                  name="Burning"))
    fig_tree.add_trace(go.Scatter(x=steps, y=forest.burned_trees_hist,
                                  name="Burned"))
    fig_tree.update_layout(title="Tree counts",
                           legend=dict(orientation="h"))

    # -------- advanced metrics time series ------------------------------
    fig_met = go.Figure()
    fig_met.add_trace(go.Scatter(x=steps, y=forest.entropy_hist,
                                 name="Entropy"))
    fig_met.add_trace(go.Scatter(x=steps, y=forest.R0_hist,
                                 name="R₀"))
    fig_met.add_trace(go.Scatter(x=steps, y=forest.tau_hist,
                                 name="τ̂"))
    fig_met.add_trace(go.Scatter(x=steps, y=forest.largest_cluster_hist,
                                 name="Largest cluster"))
    fig_met.add_trace(go.Scatter(x=steps, y=forest.mean_cluster_hist,
                                 name="Mean cluster"))
    fig_met.update_layout(title="Metrics", legend=dict(orientation="h"))

    # equilibrium badge
    equil = "✅ Equilibrated" if forest.equil_hist[-1] else "⏳ Not equilibrated"
    return (
        make_board_fig(board), fig_cluster, fig_fire,
        fig_tree, fig_met, equil
    )

# ───────────────────────── Pause / Play ─────────────────────────────────
@app.callback(
    Output("interval", "disabled"),
    Input("btn-toggle", "n_clicks"),
    State("interval", "disabled"),
    prevent_initial_call=True,
)
def toggle_running(_, disabled):
    return not disabled

# ───────────────────────── Reset simulation ────────────────────────────
@app.callback(
    Output("_dummy", "children"),
    Output("forest-graph", "figure", allow_duplicate=True),
    Input("btn-reset", "n_clicks"),
    State("width","value"), State("height","value"),
    State("p-tree","value"),
    State("p-grow","value"), State("f-light","value"),          # NEW
    State("density","value"), State("noise-oct","value"),
    State("wind-speed","value"), State("wind-dir","value"),
    State("rad-decay","value"), State("ign-base","value"),      # NEW
    State("max-dist","value"),  State("spot-prob","value"),     # NEW
    State("spot-range","value"),State("seed","value"),          # NEW
    prevent_initial_call=True,
)
def reset_sim(_, W,H, p_tree, p_grow, f_light, density, noise_oct,
              wind_speed, wind_dir, rad_decay, ign_base, max_dist,
              spot_prob, spot_range, seed_input):
    global forest
    wdx, wdy = map(float, wind_dir.split(","))
    seed = int(seed_input) if seed_input is not None else secrets.randbelow(1_000_000)

    with forest_lock:
        forest = Forest(
            (int(W), int(H)),
            p_tree      = float(p_tree),
            p_grow      = float(p_grow),
            f_lightning = float(f_light),
            density       = int(density),
            noise_octaves = int(noise_oct),
            wind_speed    = float(wind_speed),
            wind_dir      = (wdx, wdy),
            radiant_decay       = float(rad_decay),
            ignition_base_prob  = float(ign_base),
            max_ignition_distance = int(max_dist),
            spotting_prob       = float(spot_prob),
            spotting_range      = int(spot_range),
            seed          = seed,
        )
        board = forest.board.copy()

    return f"reset @ seed {seed}", make_board_fig(board)

# ───────────────────────── Entrypoint ──────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
