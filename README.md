# pivot-bogie-skid-steer-sim

A MuJoCo simulation study of a four-wheel-drive waste-collection robot that has **no steering motor**.
The two front wheels are fixed and driven. The two rear wheels sit on a bogie that turns freely on a
slewing bearing. The robot steers by running the front wheels at different speeds, and the rear
wheels are told how fast to go so the bogie lines up with the path. An encoder on the pivot reports
the bogie angle, and a small feedback loop (the *alignment loop*) keeps the bogie aligned.

> **Simulation only.** Nothing here was tested on a real robot. Several parameters are assumptions
> and are marked as such in `params.yaml`.

## Why I built this

Ordinary skid steer drags the wheels sideways when the robot turns, and the problem gets worse as
the wheelbase gets long compared with the track. I wanted to see whether a passive pivot bogie at
the back removes most of that scrubbing, and whether it can be kept stable with one angle sensor.

## What it does

- Derives the kinematics of the vehicle and the condition that keeps the bogie aligned.
- Compares a **locked pivot**, a **free pivot with no feedback**, and a **free pivot with the alignment loop**.
- Tracks a straight line, a circle and a figure-eight with two path controllers (a cascaded PID and a nonlinear law) that share the same alignment loop.
- Tests how robust it is: grip, mass, centre of mass, motor mismatch, sensor noise, pivot friction, speed, slope, steps, delay, encoder resolution, solver settings, reverse driving and start/stop.
- Finds the failure boundaries instead of only showing cases that work.

## Main findings (all in simulation)

- Without feedback, the free pivot jackknifes. The pivot is open-loop unstable going forward.
- With the alignment loop (k_theta = 6), the pivot stays within 0.03 deg of the commanded angle on the test path.
- Nominal tracking error is 5.5 to 12.2 mm RMS at 0.5 m/s.
- Results barely change with grip (mu 0.2 to 0.8), mass (100 to 150 kg), motor mismatch of 20 percent, or sensor noise up to 50 mm and 2.9 deg.
- It fails when pivot friction reaches about 10 N m, on cross-slopes of 8 deg with the assumed 6.1 N m motor torque, and on steps of 60 mm or more.
- The smallest circle tracked at the front-axle midpoint was 0.7 m radius.

See `data/` for every number behind these statements.

## Setup

Tested on Ubuntu with Python 3.12.

```bash
git clone https://github.com/Udhith/pivot-bogie-skid-steer-sim.git
cd pivot-bogie-skid-steer-sim
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run it

Check that the model builds and the kinematics are consistent:

```bash
python tests/check_mujoco.py
python tests/check_kinematics.py
```

Watch the robot in the 3D MuJoCo viewer. The blue line is the planned path and the red line is where
the robot actually went:

```bash
python scripts/view_sim.py --kind eight --T 60
python scripts/view_sim.py --kind circle --T 30
```

Reproduce the results (each script writes CSV files into `data/`):

```bash
python scripts/run_openloop.py       # locked vs free pivot, with and without feedback
python scripts/run_tracking.py       # straight, circle, figure-eight, PID and nonlinear
python scripts/run_robustness.py     # one-at-a-time robustness tests
python scripts/run_limits.py         # failure limits
python scripts/run_seeds.py          # noise repeats
python scripts/run_gates.py          # kinematic model vs MuJoCo comparison
python scripts/run_speed.py          # speed limits
python scripts/extra_tests.py        # solver, delay, slope, steps, reverse, start/stop and more
python scripts/make_figures.py       # redraw the plots from the saved CSV files
```

The extra tests take a few minutes on a laptop.

## Layout

```
params.yaml        every parameter, tagged with its source (cad, datasheet, assumed, derived)
sim/               model builder, controllers, trajectories, metrics
scripts/           experiments, the 3D viewer and the figure code
tests/             sanity checks for the model
tools/             reads geometry out of the STEP file (cad_to_params.py)
cad/               the STEP model and the geometry extracted from it
data/              CSV results and logs
figs/              figures
```

## Parameters you should know are assumed

Geometry comes from my CAD model and the wheel data from the hub-motor datasheet. These are **not**
measured and are only estimates or assumptions:

- Motor torque limit, 6.1 N m (continuous power divided by speed, not a datasheet value).
- Pivot friction (0.5 N m Coulomb, 0.2 N m s/rad viscous).
- Total mass split between chassis and bogie, and the centre-of-mass position.
- The footpath turning requirement (front-midpoint radius of 1.0 m or less).
- Tyre grip, and no suspension in the model.

Change them in `params.yaml` and re-run to see how the results move.

## Limitations

- One physics model (MuJoCo) only. The no-slip model and the contact model are not compared against a second simulator.
- No hardware validation.
- The standstill mode was added after seeing the creep problem at restart, so it is a fix tuned on those cases.
- Controller gains were set by hand and are the same for every test. They are not optimised.

## Paper

The method and results are written up in a paper (draft). Link to be added.

## Licence

MIT. See `LICENSE`.
