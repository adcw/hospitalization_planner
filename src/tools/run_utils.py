import os


def get_run_path(path: str):
    abs_path = os.path.abspath(path)

    os.makedirs(path, exist_ok=True)

    contents = [os.path.join(abs_path, f) for f in os.listdir(abs_path)]
    run_dirs = [d for d in contents if os.path.isdir(d)]

    curr_index = 1
    max_index = 0
    if len(run_dirs) > 0:
        for r in run_dirs:
            try:
                n = int(r.split("\\")[-1].split("_")[-1])
                if n > max_index:
                    max_index = n
            except ValueError:
                pass
        curr_index = max_index + 1

    return os.path.join(abs_path, f"run_{curr_index}")
