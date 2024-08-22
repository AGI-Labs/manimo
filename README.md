# ManiMo

A modular interface for robotic manipulation:
1. supports a suite of sensors (camera sensors, force sensors, touch sensors, audio sensors) and actuators (robotic arms: panda, and end-effectors).
2. allows users to compose different sensor modalities and actuators to form new manipulation environments. (mujoco compatible envs)
3. allows users to collect demonstrations from the created manipulation environments through
  a. teleoperation (VR, space mouse)
  b. manual control

## Prerequisites

- Install `mamba`

## Setup Instructions

### Manimo setup

1. Clone the repo from [https://github.com/AGI-Labs/manimo](https://github.com/AGI-Labs/manimo)
2. Set `MANIMO_PATH` as an environment variable in the `.bashrc` file:
   ```bash
   export MANIMO_PATH={FOLDER_PATH_TO_MANIMO}/manimo/manimo
   ```
3. Run the setup script on the client computer. Note that `mamba` setup does not work, always use `miniconda`:
   ```bash
   source setup_manimo_env_client.sh
   ```
4. Run the setup script on the server computer. Note that `mamba` setup does not work, always use `miniconda`:
   ```bash
   source setup_manimo_env_server.sh
   ```

To verify that the installation works, run the polymetis server on NUC by running the following script under the scripts folder:
```bash
python get_current_position.py
```

### Teleop VR Setup

- Install `oculus_reader` VR client:
  - follow the instructions [here](https://github.com/rail-berkeley/oculus_reader).
    
- Enable developer mode on the Oculus Quest. Follow the instructions at [https://developer.oculus.com/documentation/native/android/mobile-device-setup/](https://developer.oculus.com/documentation/native/android/mobile-device-setup/).

- Install Android ADB tools to communicate with the headset:
  ```bash
  sudo apt install android-tools-adb
  ```

- (Optional) Set up Wi-Fi access to the device using the instructions provided at [https://developer.oculus.com/documentation/native/android/ts-adb/](https://developer.oculus.com/documentation/native/android/ts-adb/).

### Zed Camera SDK Install

- Download the Zed SDK based on the CUDA driver version on your system from [https://www.stereolabs.com/developers/release](https://www.stereolabs.com/developers/release).

## Future:

1. supports on-board calbiration of different sensors.
  
## Acknowledgements:
manimo's design is heavily inspired by [franka_demo](https://github.com/AGI-Labs/franka_demo/tree/dmanus_devel)

## Projects using ManiMo:
- [An Unbiased Look at Datasets for Visuo-Motor Pre-Training](https://data4robotics.github.io/): Sudeep Dasari, Mohan Kumar Srirama, Unnat Jain, Abhinav Gupta
- [PlayFusion: Skill Acquisition via Diffusion from Language-Annotated Play](https://play-fusion.github.io/): Lili Chen, Shikhar Bahl, Deepak Pathak
- [Hearing Touch: Audio-Visual Pretraining for Contact-Rich Manipulation](https://sites.google.com/view/hearing-touch): Jared Mejia, Victoria Dean, Tess Hellebrekers, Abhinav Gupta
- [HRP: Human Affordances for Robotic Pre-Training](https://hrp-robot.github.io/): Mohan Kumar Srirama, Sudeep Dasari*, Shikhar Bahl*, Abhinav Gupta*
