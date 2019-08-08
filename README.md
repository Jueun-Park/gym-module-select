# gym-module-select

Gym environment for the autonomous car control module selector based on the reinforcement learning agent.

## Details

### Observation

Raw image from the cam which see the front of the car. The shape is `(80, 160, 3)`.

### Action

* `0` to select default lane tracking module
* `1` to select SAC RL agent module

### Reward

`(original env reward) - (one frame processing time (ms))`

> `original env reward`: Please refer the section that name is _Reward Function: Go Fast but Stay on the Track!_ in [araffin's blog post](https://towardsdatascience.com/learning-to-drive-smoothly-in-minutes-450a7cdb35f4)

## Credits

- [Laurent Desegur](https://medium.com/@ldesegur/a-lane-detection-approach-for-self-driving-vehicles-c5ae1679f7ee) for lane detection module
- [araffin](https://github.com/araffin/learning-to-drive-in-5-minutes) for VAE-SAC agent module
- [Tawn Kramer](https://github.com/tawnkramer) for Donkey simulator and Donkey Gym.
- [Stable-Baselines](https://github.com/hill-a/stable-baselines) for RL agent implementations.
