import os
import json

def fix(run_path):
    config_json_path = os.path.join(run_path, "config.json")

    with open(config_json_path, "r") as f:
        config = json.load(f)

    del config["negative_ratio"]

    new_config_path = os.path.join(
        os.path.dirname(config_json_path), "config_nonegratio.json"
    )
    with open(new_config_path, "w") as f:
        json.dump(config, f, sort_keys=True, indent=2)


if __name__ == "__main__":
    import sys
    run_path = sys.argv[1]
    fix(run_path)
