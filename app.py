import dash
import numpy as np
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output, State
from flask import Flask

# Constants
g = 9.81
D = 0.15  # Circle diameter in meters


def shot_time_spacing(D, v, theta):
    theta_rad = np.radians(theta)
    v_x = v * np.cos(theta_rad)
    v_y = v * np.sin(theta_rad)
    t_collision = (
        v_y - np.sqrt(2) * np.sqrt(np.sqrt(v_x**4 + g**2 * D**2) - v_x**2)
    ) / g
    t_peak = v_y / g
    return t_peak - t_collision


def shots_per_second(D, v, theta):
    spacing = shot_time_spacing(D, v, theta)
    return 1 / spacing


def calculate_trajectory(v, theta_deg):
    theta = np.radians(theta_deg)
    v_x = v * np.cos(theta)
    v_y = v * np.sin(theta)
    t_flight = 2 * v_y / g
    max_x = v_x * t_flight
    max_y = (v_y**2) / (2 * g)
    return t_flight, max_x, max_y, v_x, v_y


def generate_frame_data(v, theta, bps, v_std=0, theta_std=0, use_gaussian=False):
    """Generate positions for all circles at current time"""
    interval = 1 / bps
    t_flight, max_x, max_y, vx, vy = calculate_trajectory(v, theta)

    # Calculate how many circles should be visible
    num_circles = int(t_flight / interval) + 1

    x_positions = []
    y_positions = []
    colors = []

    circles_data = []

    for i in range(num_circles):
        t_rel = i * interval

        if use_gaussian:
            # Sample with Gaussian distribution
            v_sample = v + np.clip(np.random.normal(0, v_std), -v_std, v_std)
            theta_sample = theta + np.clip(
                np.random.normal(0, theta_std), -theta_std, theta_std
            )
            _, _, _, vx_i, vy_i = calculate_trajectory(v_sample, theta_sample)
            t_flight_i = 2 * vy_i / g
        else:
            vx_i, vy_i = vx, vy
            t_flight_i = t_flight

        if t_rel <= t_flight_i:
            x = vx_i * t_rel
            y = vy_i * t_rel - 0.5 * g * t_rel**2

            circles_data.append({"x": x, "y": y, "vx": vx_i, "vy": vy_i})

    # Check for collisions
    overlapping = set()
    for i in range(len(circles_data)):
        for j in range(i + 1, len(circles_data)):
            dist_sq = (circles_data[i]["x"] - circles_data[j]["x"]) ** 2 + (
                circles_data[i]["y"] - circles_data[j]["y"]
            ) ** 2
            if dist_sq <= D**2:
                overlapping.add(i)
                overlapping.add(j)

    # Build positions and colors
    for i, circle in enumerate(circles_data):
        x_positions.append(circle["x"])
        y_positions.append(circle["y"])
        colors.append("red" if i in overlapping else "white")

    return x_positions, y_positions, colors, max_x, max_y


# Initialize Flask server
server = Flask(__name__)

# Initialize Dash app
app = dash.Dash(__name__, server=server)

app.layout = html.Div(
    [
        html.H1("Midair Ballistic Collision Calculator", style={"textAlign": "center"}),
        html.Div(
            [
                html.H3(
                    "Shared Controls",
                    style={"textAlign": "center", "marginBottom": "20px"},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Velocity (m/s):"),
                                dcc.Slider(
                                    id="velocity-slider",
                                    min=1,
                                    max=30,
                                    step=0.5,
                                    value=10,
                                    marks={i: str(i) for i in range(0, 31, 5)},
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                ),
                            ],
                            style={
                                "width": "45%",
                                "display": "inline-block",
                                "marginRight": "5%",
                            },
                        ),
                        html.Div(
                            [
                                html.Label("Angle (degrees):"),
                                dcc.Slider(
                                    id="angle-slider",
                                    min=0,
                                    max=90,
                                    step=1,
                                    value=45,
                                    marks={i: str(i) for i in range(0, 91, 15)},
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                ),
                            ],
                            style={"width": "45%", "display": "inline-block"},
                        ),
                    ],
                    style={"marginBottom": "20px"},
                ),
                html.Div(
                    [
                        html.Label("Balls Per Second (BPS):"),
                        dcc.Slider(
                            id="bps-slider",
                            min=0.1,
                            max=100,
                            step=0.1,
                            value=10,
                            marks={i: str(i) for i in [0.1, 1, 10, 25, 50, 75, 100]},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ],
                    style={"marginBottom": "20px"},
                ),
                html.Div(
                    [
                        html.H4(
                            id="critical-bps-label",
                            style={"color": "red", "textAlign": "center"},
                        )
                    ],
                    style={"marginBottom": "20px"},
                ),
                html.Hr(),
                html.H3(
                    "Gaussian Variation Controls (applies to bottom plot only)",
                    style={
                        "textAlign": "center",
                        "marginTop": "20px",
                        "marginBottom": "20px",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Velocity Std Dev:"),
                                dcc.Slider(
                                    id="v-std-slider",
                                    min=0,
                                    max=2,
                                    step=0.05,
                                    value=0.25,
                                    marks={i / 2: str(i / 2) for i in range(0, 5)},
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                ),
                            ],
                            style={
                                "width": "45%",
                                "display": "inline-block",
                                "marginRight": "5%",
                            },
                        ),
                        html.Div(
                            [
                                html.Label("Angle Std Dev:"),
                                dcc.Slider(
                                    id="theta-std-slider",
                                    min=0,
                                    max=10,
                                    step=0.1,
                                    value=1.0,
                                    marks={i: str(i) for i in range(0, 11, 2)},
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                ),
                            ],
                            style={"width": "45%", "display": "inline-block"},
                        ),
                    ],
                    style={"marginBottom": "30px"},
                ),
            ],
            style={
                "width": "90%",
                "margin": "auto",
                "padding": "20px",
                "backgroundColor": "#f0f0f0",
                "borderRadius": "10px",
            },
        ),
        html.Div(
            [
                html.H2(
                    "Ideal Trajectories (No Variation)",
                    style={"textAlign": "center", "marginTop": "30px"},
                ),
                dcc.Graph(id="ideal-plot", style={"height": "500px"}),
            ]
        ),
        html.Div(
            [
                html.H2(
                    "Trajectories with Gaussian Variation",
                    style={"textAlign": "center", "marginTop": "30px"},
                ),
                dcc.Graph(id="gaussian-plot", style={"height": "500px"}),
            ]
        ),
    ]
)


@app.callback(
    [Output("bps-slider", "value"), Output("critical-bps-label", "children")],
    [Input("velocity-slider", "value"), Input("angle-slider", "value")],
    prevent_initial_call=True,
)
def update_critical_bps(velocity, angle):
    crit_bps = shots_per_second(D, velocity, angle)
    return crit_bps, f"Critical BPS: {crit_bps:.2f}"


@app.callback(
    Output("ideal-plot", "figure"),
    [
        Input("velocity-slider", "value"),
        Input("angle-slider", "value"),
        Input("bps-slider", "value"),
    ],
)
def update_ideal_plot(velocity, angle, bps):
    x_pos, y_pos, colors, max_x, max_y = generate_frame_data(
        velocity, angle, bps, v_std=0, theta_std=0, use_gaussian=False
    )

    # Create circle shapes
    shapes = []
    for i in range(len(x_pos)):
        shapes.append(
            {
                "type": "circle",
                "x0": x_pos[i] - D / 2,
                "y0": y_pos[i] - D / 2,
                "x1": x_pos[i] + D / 2,
                "y1": y_pos[i] + D / 2,
                "fillcolor": colors[i],
                "line": {"color": "black", "width": 1},
            }
        )

    # Create the figure
    fig = go.Figure()

    # Add invisible scatter points to set the data range
    fig.add_trace(
        go.Scatter(
            x=[0, max_x],
            y=[0, max_y * 1.2],
            mode="markers",
            marker=dict(size=0.1, color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        xaxis=dict(
            title="Distance (m)",
            range=[0, max_x if max_x > 0 else 10],
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title="Height (m)",
            range=[0, max_y * 1.2 if max_y > 0 else 10],
        ),
        shapes=shapes,
        showlegend=False,
        plot_bgcolor="white",
        xaxis_showgrid=True,
        yaxis_showgrid=True,
        margin=dict(l=50, r=50, t=20, b=50),
    )

    return fig


@app.callback(
    Output("gaussian-plot", "figure"),
    [
        Input("velocity-slider", "value"),
        Input("angle-slider", "value"),
        Input("bps-slider", "value"),
        Input("v-std-slider", "value"),
        Input("theta-std-slider", "value"),
    ],
)
def update_gaussian_plot(velocity, angle, bps, v_std, theta_std):
    x_pos, y_pos, colors, max_x, max_y = generate_frame_data(
        velocity, angle, bps, v_std, theta_std, use_gaussian=True
    )

    # Create circle shapes
    shapes = []
    for i in range(len(x_pos)):
        shapes.append(
            {
                "type": "circle",
                "x0": x_pos[i] - D / 2,
                "y0": y_pos[i] - D / 2,
                "x1": x_pos[i] + D / 2,
                "y1": y_pos[i] + D / 2,
                "fillcolor": colors[i],
                "line": {"color": "black", "width": 1},
            }
        )

    # Create the figure
    fig = go.Figure()

    # Add invisible scatter points to set the data range
    fig.add_trace(
        go.Scatter(
            x=[0, max_x],
            y=[0, max_y * 1.2],
            mode="markers",
            marker=dict(size=0.1, color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        xaxis=dict(
            title="Distance (m)",
            range=[0, max_x if max_x > 0 else 10],
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title="Height (m)",
            range=[0, max_y * 1.2 if max_y > 0 else 10],
        ),
        shapes=shapes,
        showlegend=False,
        plot_bgcolor="white",
        xaxis_showgrid=True,
        yaxis_showgrid=True,
        margin=dict(l=50, r=50, t=20, b=50),
    )

    return fig


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
