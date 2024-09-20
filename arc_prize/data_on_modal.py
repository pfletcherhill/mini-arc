import json
import os
import time

import modal

from arc_prize.synth_data.html import capture_html_screenshot, process_screenshot

modal_image = modal.Image.debian_slim().pip_install(["selenium", "Pillow"])
modal_app = modal.App(name="arc-generator", image=modal_image)

data_volume = modal.Volume.from_name("arc-data")
html_volume = modal.Volume.from_name("arc-html")


@modal_app.function(
    volumes={"/vol/data": data_volume, "/vol/html": html_volume},
    timeout=(60 * 60 * 12),
)
def generate_data_from_html(
    puzzle_id: str, num_puzzles: int, max_dim: int, dataset_dir: str
):
    os.makedirs(f"/vol/data/{dataset_dir}", exist_ok=True)
    html_file = f"/vol/html/{puzzle_id}.html"

    start_time = time.time()
    puzzles = []
    max_tries = num_puzzles * 5

    print("Starting", puzzle_id)

    for i in range(max_tries):
        raw_screenshot = capture_html_screenshot(html_file)
        arc_puzzle_data, _ = process_screenshot(raw_screenshot)

        puzzle_dim = 0
        for pair in arc_puzzle_data:
            for grid in pair:
                dim = max([len(grid), len(grid[0])])
                if dim > puzzle_dim:
                    puzzle_dim = dim

        if puzzle_dim <= max_dim:
            puzzles.append(arc_puzzle_data)

        duration = time.time() - start_time
        if len(puzzles) >= num_puzzles:
            print(
                "Finished", puzzle_id, "after gen", i, "with num puzzles", len(puzzles)
            )
            print(f"Total time: {duration:.2f}s ({(duration / 60):.2f}m)")
            break
        elif i % 10 == 0:
            per_puzzle_time = duration / len(puzzles)
            print(
                f"Done {i} puzzles. Time elapsed: {duration:.2f}s ({(duration / 60):.2f}m). Per puzzle: {per_puzzle_time:.2fs}s"
            )

    with open(f"/vol/data/{dataset_dir}/{puzzle_id}.json", "w") as f:
        json.dump(puzzles, f)
