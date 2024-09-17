import base64
import io
import os
from typing import Optional

import anthropic
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

load_dotenv()

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
claude_client = anthropic.Anthropic(api_key=os.environ.get("CLAUDE_API_KEY"))


COLORS = [
    "#c2c0c0",  # padding grey
    "#111111",  # black
    "#1E93FF",  # blue
    "#F93C31",  # red
    "#4FCC30",  # green
    "#FFDC00",  # yellow
    "#E6E6E6",  # grey
    "#E53AA3",  # magenta
    "#FF851B",  # orange
    "#87D8F1",  # light blue
    "#921231",  # maroon
]


def read_file(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except IOError:
        print(f"Error: There was an issue reading the file '{file_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return ""


def get_html_template() -> str:
    return read_file("html/template.html")


def get_instructions(puzzle_id: str) -> str:
    return read_file(f"prompts/{puzzle_id}.txt")


GENERAL_INSTRUCTIONS = """General Instructions:
- Change the HTML title to the name of the puzzle
- On each page load, pick a random number of pairs (between 2-5) and add more pairs using the pattern already in the document for the first two pairs. The number of pairs should be determined randomly with each page load.
- Each pair should include two containers, sized 30x30 pixels each. Each container will contain an input grid div and an output grid div, which the puzzle instructions will help you define. Do not change the size of the containers, only the sizes of the grids inside them. 
- Do not modify the CSS for main, pair, container classes. None of the classes in the template should be changed."""

WARNINGS = """Warnings:
- Do not use a canvas, because the output will be blurry.
- Make sure your script does not cause an infinite loop or cause the page to crash when the HTML is loaded.
- Just return the HTML document for saving in a file."""


def make_prompt(puzzle_id: str) -> str:
    sections = [
        f"Update this HTML document using the instructions below. Think step-by-step and make sure the HTML is correct. The name of the puzzle is {puzzle_id}.",
        get_html_template(),
        GENERAL_INSTRUCTIONS,
        f"Instructions for puzzle {puzzle_id}:\n{get_instructions(puzzle_id)}",
        WARNINGS,
    ]
    return "\n\n".join(sections)


def get_html_openai(prompt: str) -> str:
    chat_completion = openai_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o",
    )
    return chat_completion.choices[0].message.content or ""


def get_html_claude(prompt: str) -> str:
    message = claude_client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        temperature=0.2,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    )
    print(message.content)
    return message.content[0].text


def get_html(puzzle_id: str, model: str, output_file_path: Optional[str] = None) -> str:
    prompt = make_prompt(puzzle_id)
    print(prompt)
    if model == "claude":
        html = get_html_claude(prompt)
    elif model == "openai":
        html = get_html_openai(prompt)
    else:
        raise Exception("invalid model name")
    output_file_path = output_file_path or f"html/{puzzle_id}/{puzzle_id}_{model}.html"
    with open(output_file_path, "w") as f:
        f.write(html)
    return html


def capture_html_screenshot(html_file, output_size=(60, 150)):
    # Set up headless Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--hide-scrollbars")
    chrome_options.add_argument("--force-device-scale-factor=1")
    chrome_options.add_argument("--high-dpi-support=1")

    driver = webdriver.Chrome(options=chrome_options)

    driver.execute_cdp_cmd(
        "Emulation.setDeviceMetricsOverride",
        {
            "width": output_size[0],
            "height": output_size[1],
            "deviceScaleFactor": 1,
            "mobile": False,
        },
    )

    # Load the HTML file
    driver.get(f"file:///{html_file}")

    # Capture screenshot
    screenshot = driver.get_screenshot_as_base64()
    driver.quit()

    return screenshot


def hex_to_rgb(hex_color):
    return tuple(int(hex_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb_color):
    return f"#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}"


def color_distance(color1, color2):
    return sum((a - b) ** 2 for a, b in zip(color1, color2)) ** 0.5


def match_color(color, defined_colors):
    rgb_color = hex_to_rgb(color)
    distances = [color_distance(rgb_color, hex_to_rgb(c)) for c in defined_colors]
    return defined_colors[distances.index(min(distances))]


def process_screenshot(screenshot, grid_size=(2, 5), cell_size=30):
    image_data = base64.b64decode(screenshot)
    image = Image.open(io.BytesIO(image_data))
    print(image, image.size)

    grid = []
    for row in range(grid_size[1]):
        grid_row = []
        for col in range(grid_size[0]):
            left = col * cell_size
            top = row * cell_size
            right = left + cell_size
            bottom = top + cell_size

            print("crop", left, top, right, bottom)

            cell = image.crop((left, top, right, bottom))
            pixel_representation = []
            for y in range(cell_size):
                pixel_row = []
                for x in range(cell_size):
                    pixel = cell.getpixel((x, y))
                    hex_color = rgb_to_hex(pixel)
                    matched_color = match_color(hex_color, COLORS)
                    # print(f"pixel ({x}, {y}): {pixel}")
                    pixel_row.append(matched_color)
                pixel_representation.append(pixel_row)
            grid_row.append(pixel_representation)
        grid.append(grid_row)

    return grid, image


def display_images(image, grid):
    fig, axes = plt.subplots(5, 2, figsize=(10, 25))
    fig.suptitle("Grid of 30x30 Images", fontsize=16)

    for row in range(5):
        for col in range(2):
            cell_data = np.array(
                [
                    [
                        tuple(int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
                        for color in row
                    ]
                    for row in grid[row][col]
                ]
            )
            axes[row, col].imshow(cell_data.astype(np.uint8))
            axes[row, col].axis("off")
            axes[row, col].set_title(f"Cell ({row+1}, {col+1})")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Display full screenshot
    plt.figure(figsize=(10, 25))
    plt.imshow(image)
    plt.axis("off")
    plt.title("Full Screenshot")
    plt.show()
