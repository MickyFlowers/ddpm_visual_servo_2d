from omni.isaac.kit import SimulationApp

app_config = {"headless": True}
simulation_app = SimulationApp(app_config)
import os

from env.isaac_env import env


def main():
    root_path = os.path.dirname(os.path.realpath(__file__))

    isaac_env = env(root_path=root_path, render=True, physics_dt=1 / 60.0)
    while simulation_app.is_running():
        simulation_app.update()
        isaac_env.run()
    simulation_app.close()


if __name__ == "__main__":
    main()
