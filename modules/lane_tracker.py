import numpy as np
from simple_pid import PID
from modules.lane_detector import LaneDetector
from modules.add_delay import add_delay

EMERGENCY_MODE = True

class LaneTracker:
    def __init__(self, delay_weight, static_term):
        self.delay_weight = delay_weight
        self.static_term = static_term
        self.steer_controller = PID(Kp=2.88,
                                Ki=0.0,
                                Kd=0.0818,
                                output_limits=(-1, 1),
                                )
        self.base_speed = 1
        self.speed_controller = PID(Kp=1.0,
                                Ki=0.0,
                                Kd=0.125,
                                output_limits=(-1, 1),
                                )

        self.detector = LaneDetector()

    def predict(self, raw_obs, proc_state):
        add_delay(proc_state, self.delay_weight, self.static_term)
        is_done, angle_error = self.detector.detect_lane(raw_obs)
        if is_done or EMERGENCY_MODE:
            angle_error = -angle_error
            steer = self.steer_controller(angle_error)
            reduction = self.speed_controller(steer)
            speed = self.base_speed - np.abs(reduction)
        else:
            angle_error = 0
            steer = self.steer_controller(angle_error)
            speed = self.base_speed

        return [[steer, speed]]
