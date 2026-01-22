# Midair Ballistic Collision Calculator - Web App

This is a Flask/Dash web application for visualizing ballistic trajectories and detecting midair collisions.

- Shows **both** ideal and Gaussian trajectories simultaneously
- Top plot: Ideal trajectories (no variation)
- Bottom plot: Trajectories with Gaussian variation
- Shared controls for velocity, angle, and BPS
- Separate controls for Gaussian variation (only affect bottom plot)

## Features

- **Real-time interactive controls** - Sliders update the visualization immediately
- **Collision Detection:** Overlapping projectiles turn red
- **Critical BPS Display:** Shows the theoretical maximum rate before collisions
- **Side-by-side comparison** (dual plot version) of ideal vs. realistic trajectories

## Installation

1. Install Python 3.8 or higher

2. Install required packages:

```bash
pip install -r requirements.txt
```

## Running the App

```bash
python midair_collision_dual_app.py
```

The app will start on `http://localhost:8050`

Open your browser and navigate to that address.

## Usage

### Controls

**Shared Controls (affect both plots):**

- **Velocity:** Set the initial velocity (m/s)
- **Angle:** Set the launch angle (degrees)
- **BPS:** Balls per second - how frequently projectiles are launched
- **Critical BPS:** (Red text) Shows the maximum theoretical rate

**Gaussian Variation Controls (affect bottom plot only):**

- **Velocity Std Dev:** Variation in velocity
- **Angle Std Dev:** Variation in angle

### How it Works

The app calculates ballistic trajectories and displays circles at regular intervals. When you change the velocity or angle, the "Critical BPS" automatically updates to show the maximum rate at which projectiles can be fired without colliding in midair.

- **White circles:** No collision
- **Red circles:** Collision detected (circles overlap)

The dual plot version lets you **compare** ideal conditions (top) vs. realistic conditions with variation (bottom) side-by-side.

## Notes

- The animation updates automatically based on an interval timer
- Adjusting sliders will regenerate the trajectories in real-time
- The Gaussian mode adds realistic variation to each projectile's path
