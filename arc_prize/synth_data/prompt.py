import base64
import io
import os
from typing import Optional

import anthropic
import matplotlib.colors as mcolors
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

PADDING_COLOR = COLORS[0]


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
- Do not modify the CSS for main, pair, container classes. None of the classes in the template should be changed.
- Pick a background color for all the grids, which should be black for 60% of puzzles"""

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


def display_images(image, grid):
    fig, axes = plt.subplots(5, 2, figsize=(10, 25))
    fig.suptitle("Grid of 30x30 Images", fontsize=16)

    cmap = mcolors.ListedColormap(COLORS[1:])
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            print(i, j, cell)
            axes[i, j].imshow(cell, cmap=cmap, vmin=0, vmax=(len(COLORS) - 2))
            axes[i, j].axis("off")
            axes[i, j].set_title(f"Cell ({i+1}, {j+1})")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Display full screenshot
    plt.figure(figsize=(10, 25))
    plt.imshow(image)
    plt.axis("off")
    plt.title("Full Screenshot")
    plt.show()
