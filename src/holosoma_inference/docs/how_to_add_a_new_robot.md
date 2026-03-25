# Adding a New Robot to Holosoma Inference

This document is meant to serve as a high-level overview on how to add a new robot to work with `holosoma_inference`

---

### 1. Subclass `BaseInterface`

Create `sdk/myrobot/myrobot_interface.py` and implement all abstract methods. Most importantly, `get_low_state()` and  `send_low_command()`:

```python
from holosoma_inference.sdk.base.base_interface import BaseInterface

class MyRobotInterface(BaseInterface):

    def __init__(self, robot_config, domain_id=0, interface_str=None, use_joystick=True):
        super().__init__(robot_config, domain_id, interface_str, use_joystick)
        # Initialize your SDK / communication layer here

    def get_low_state(self) -> np.ndarray:
        """Return shape (1, 3+4+N+3+3+N) array:
        [base_pos(3), quat(4), joint_pos(N), lin_vel(3), ang_vel(3), joint_vel(N)]
        """
        ...

    def send_low_command(self, cmd_q, cmd_dq, cmd_tau,
                         dof_pos_latest=None, kp_override=None, kd_override=None):
        """Map joint-space commands to your robot's motor API."""
        ...
```

Notes:
- `get_low_state()` must return a `(1, 2*(3+N)+4+3)` numpy array in the exact field order above. The policy reads this array by offset, so the layout matters.
- `send_low_command()` receives arrays of length `num_joints` (joint-space). You are responsible for remapping to motor-space if they differ.
- Gain properties (`kp_level`/`kd_level`) are float multipliers (default 1.0) that scale `motor_kp`/`motor_kd` from the config.


### 2. Implement `BaseSdk2Bridge`


### 3. [Optional] Register entrypoint
