import argparse
import os
import sys
from importlib.machinery import SourceFileLoader
from ycy_utils.common_utils import seed_everything
from posedetector import PoseDetector

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    seed_everything(seed=experiment.config["seed"])

    if "scene_path" not in experiment.config:
        results_dir = os.path.join(
            experiment.config["workdir"], experiment.config["run_name"]
        )
        scene_path = os.path.join(results_dir, "params.npz")
    else:
        scene_path = experiment.config["scene_path"]
    viz_cfg = experiment.config["viz"]

    detector = PoseDetector(scene_path, viz_cfg)

    detector.run()


if __name__ == '__main__':
    main()
