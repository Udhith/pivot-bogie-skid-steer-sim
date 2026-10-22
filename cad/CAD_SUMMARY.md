# CAD-derived parameters (from Simulation_Design_of_Chassis_and_Steering.step)
Frame: X along the robot (pivot at x=0, fixed wheels at x=-1133.5 mm), Y width, Z up. Units mm in CAD, m in params.yaml.
| Quantity | Value | How obtained |
|---|---|---|
| Wheel radius (mean of 4) | 0.1326 m (0.130-0.135 spread) | bounding box of wheel solids |
| Pivot to fixed (front) axle, l | 1.1335 m | x of bearing vs fixed wheels |
| Fixed-wheel track | 0.634 m | y between wheel centres |
| Bogie-wheel track | 0.320 m | y between wheel centres |
| Trail e (axle behind pivot) | 0.000 m (+-0.002) | bogie axle x vs bearing x |
| Bearing height above bogie axle | 176 mm | z of bearing disc vs axle |
| Bearing height above fixed axle | 143 mm | z of bearing disc vs axle |
| Bogie axle lower than fixed axle | 33 mm | z of wheel centres |
Not available from CAD: masses of motors, battery, conveyor, payload; CoM; pivot friction; real tyre width.
