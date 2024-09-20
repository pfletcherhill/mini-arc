import base64
import io

from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

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


def get_web_driver(width: int, height: int):
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
            "width": width,
            "height": height,
            "deviceScaleFactor": 1,
            "mobile": False,
        },
    )

    return driver


def capture_html_screenshot(driver, html_file):
    driver.get(f"file:///{html_file}")
    screenshot = driver.get_screenshot_as_base64()

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

    grid = []
    for row in range(grid_size[1]):
        grid_row = []
        for col in range(grid_size[0]):
            left = col * cell_size
            top = row * cell_size
            right = left + cell_size
            bottom = top + cell_size

            cell = image.crop((left, top, right, bottom))
            pixel_representation = []
            for y in range(cell_size):
                pixel_row = []
                for x in range(cell_size):
                    pixel = cell.getpixel((x, y))
                    hex_color = rgb_to_hex(pixel)
                    matched_color = match_color(hex_color, COLORS)
                    matched_color_i = list.index(COLORS, matched_color)
                    if matched_color_i > 0:
                        pixel_row.append(matched_color_i - 1)
                if len(pixel_row) > 0:
                    pixel_representation.append(pixel_row)
            if len(pixel_representation) > 0:
                grid_row.append(pixel_representation)
        if len(grid_row) > 0:
            grid.append(grid_row)

    return grid, image
