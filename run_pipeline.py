"""
Master pipeline runner.

Usage:
  python run_pipeline.py            # run all 4 steps
  python run_pipeline.py --from 2   # resume from step 2
  python run_pipeline.py --only 3   # run step 3 only
  python run_pipeline.py --test     # quick test: 3 clusters, 10 galaxies each
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def patch_for_test():
    """Temporarily lower limits for a quick smoke-test."""
    import config
    config.MAX_CLUSTERS               = 3
    config.MAX_GALAXIES_PER_CLUSTER   = 10
    config.PROJ_AXES_OVERRIDE         = ["z"]
    print("TEST MODE: 3 clusters × 10 galaxies = up to 30 images\n")


def run_step(n):
    if n == 1:
        from importlib import import_module
        m = import_module("01_find_clusters")
        m.find_clusters()
    elif n == 2:
        from importlib import import_module
        m = import_module("02_extract_galaxy_list")
        m.extract_galaxies()
    elif n == 3:
        from importlib import import_module
        m = import_module("03_generate_images")
        m.generate_images()
    elif n == 4:
        from importlib import import_module
        m = import_module("04_quality_check")
        m.run_qc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from",  dest="from_step", type=int, default=1)
    parser.add_argument("--only",  dest="only_step", type=int, default=None)
    parser.add_argument("--test",  action="store_true")
    args = parser.parse_args()

    if args.test:
        patch_for_test()

    steps = [args.only_step] if args.only_step else range(args.from_step, 5)

    for step in steps:
        print(f"\n{'='*60}")
        print(f"  STEP {step}")
        print(f"{'='*60}\n")
        run_step(step)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
