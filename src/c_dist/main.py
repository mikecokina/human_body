"""
Color distance based on paper https://www.compuphase.com/cmetric.htm.
"""
import abc
import enum
import os
import random
from math import sqrt
from pathlib import Path
from typing import Tuple

import PIL
import numpy as np
import pandas as pd
from PIL import Image

from colormath.color_objects import sRGBColor, LabColor, xyYColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

from matplotlib import pyplot as plt

from c_dist.kmeans import clab_kmeans

THIS_DIR = Path(__file__).parent
COLORS_DIR = THIS_DIR / 'data'
COLORS_LAB_CENTRES = COLORS_DIR / 'lab_cie_centres.npy'
COLORS_DIR_ASSIGNED = COLORS_DIR / 'assigned'
HTML_COLORS = THIS_DIR / 'html_colors.csv'
COLORS_DIST_TOL = 10


class AbstractColor(metaclass=abc.ABCMeta):
    def __init__(self, hexa):
        self.hexa: str = hexa
        self.rgb: Tuple = self.hex_to_rgb(hexa)
        self.rgb_decimal: Tuple = tuple([value / 255.0 for value in self.rgb])

    @property
    def r(self):
        return self.rgb[0]

    @property
    def g(self):
        return self.rgb[1]

    @property
    def b(self):
        return self.rgb[2]

    @staticmethod
    def hex_to_rgb(hexa: str) -> Tuple[int]:
        """
        Convert hex value to rgb value (0 - 255)
        """
        hexa = hexa[1:] if hexa.startswith('#') else hexa
        return tuple(int(hexa[i:i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def rgb_to_hex(rgb: Tuple[int]) -> str:
        return f'#{rgb[0] << 16 | rgb[1] << 8 | rgb[2]:06x}'

    @classmethod
    def as_random_rgb(cls):
        return cls('#' + ''.join(["%02x" % random.randint(0, 0xFF) for _ in range(3)]))
        # return cls("#" + ''.join([random.choice('abcdef0123456789') for _ in range(6)]))


class LAB(AbstractColor):
    def __init__(self, hexa: str):
        super(LAB, self).__init__(hexa)
        self.lab = self.rgb_to_lab(self.rgb)

    def __str__(self):
        return f'Lab__l:{self.lab.lab_l}-a:{self.lab.lab_a}-b:{self.lab.lab_b}'

    @property
    def l(self):  # noqa
        return self.lab.lab_l

    @property
    def a(self):
        return self.lab.lab_a

    @property
    def b(self):
        return self.lab.lab_b

    @staticmethod
    def rgb_to_lab(rgb):
        srgb = sRGBColor(rgb[0] / 255., rgb[1] / 255., rgb[2] / 255.)
        return convert_color(srgb, LabColor)

    @staticmethod
    def lab_to_rgb(lab: LabColor) -> Tuple:
        srgb = convert_color(lab, sRGBColor)
        return int(srgb.rgb_r * 255), int(srgb.rgb_g * 255), int(srgb.rgb_b * 255)

    def delta_e_distance(self, other: 'LAB'):
        return delta_e_cie2000(self.lab, other.lab)

    def distance(self, other):
        return self.delta_e_distance(other)

    def assing_to_palette(self, palette):
        palette = palette.values()
        deltas = [self.distance(self.__class__(color)) for color in palette]
        condition = [delta_e < COLORS_DIST_TOL for delta_e in deltas]

        if any(condition):
            argmin = np.argmin(deltas)
            return palette[argmin]
        return None


class Yxy(AbstractColor):
    def __init__(self, hexa: str):
        super(Yxy, self).__init__(hexa)
        self.yxy = self.rgb_to_yxy(self.rgb)

    @property
    def Y(self):  # noqa
        return self.yxy.xyy_Y

    @property
    def x(self):
        return self.yxy.xyy_x

    @property
    def y(self):
        return self.yxy.xyy_y

    @staticmethod
    def rgb_to_yxy(rgb):
        srgb = sRGBColor(rgb[0] / 255., rgb[1] / 255., rgb[2] / 255.)
        return convert_color(srgb, xyYColor)

    def __str__(self):
        return f"Yxy__Y:{self.Y}-x:{self.x}-y:{self.y}"


class RGB(AbstractColor):
    def __init__(self, hexa: str):
        super(RGB, self).__init__(hexa)

    def __str__(self):
        return f"RGB__{self.hexa}"

    def redmean_distance(self, other: 'RGB'):
        rmean = int((self.r + other.r) / 2)
        r = int(self.r - other.r)
        g = int(self.g - other.g)
        b = int(self.b - other.b)
        return sqrt((((512 + rmean) * r * r) >> 8) + 4 * g * g + (((767 - rmean) * b * b) >> 8))

    def distance(self, other):
        return self.redmean_distance(other)

    def assing_to_palette(self, palette):
        palette = palette.values()
        deltas = [self.distance(self.__class__(color)) for color in palette]
        condition = [delta_e < COLORS_DIST_TOL for delta_e in deltas]

        if any(condition):
            argmin = np.argmin(deltas)
            return palette[argmin]
        return None


class NaiveRGB(RGB):
    def distance(self, other: 'NaiveRGB'):
        return abs(self.r - other.r) + abs(self.g - other.g) + abs(self.b - other.b)

    def assing_to_palette(self, palette):
        palette = palette.values()
        argmin = np.argmin([self.distance(self.__class__(color)) for color in palette])
        return palette[argmin]


class AbstractEnum(enum.Enum):
    @classmethod
    def values(cls):
        """Returns a list of all the enum values."""
        return list(cls._value2member_map_.keys())


class CustomPalette(AbstractEnum):
    white = "#ffffff"
    black = "#000000"

    blue = "#1f77b4"
    orange = "#ff7f0e"
    green = "#2ca02c"
    red = "#d62728"
    purple = "#9467bd"
    brown = "#8c564b"
    pink = "#e377c2"
    gray = "#7f7f7f"
    olive = "#bcbd22"
    cyan = "#17becf"


class CIE2000Palette(AbstractEnum):
    green = '#00a347'
    yellowish_green = '#aad13c'
    yellow_green = '#b9d604'
    greenish_yellow = '#ebe900'
    yellow = '#eae75e'
    yellowish_orange = '#e7e000'
    orange = '#e4b81d'
    orange_pink = '#f0cca2'
    reddish_orange = '#d87733'
    red = '#bf1b4b'
    pink = '#f5dcd0'
    purplish_red = '#d14188'
    purplish_pink = '#f3d0db'
    red_purple = '#af2384'
    reddish_purple = '#c4408f'
    purple = '#f6559e'
    bluish_purple = '#5c66b1'
    purplish_blue = '#5879bf'
    blue = '#5c8aca'
    greenish_blue = '#6eafc7'
    bluegreen = '#5fa4be'
    bluish_green = '#18a279'


class CIE2000PaletteExtended(AbstractEnum):
    black = '#000000'
    white = '#ffffff'
    gray = '#bcbebc'

    green = '#00a347'
    yellowish_green = '#aad13c'
    yellow_green = '#b9d604'
    greenish_yellow = '#ebe900'
    yellow = '#eae75e'
    yellowish_orange = '#e7e000'
    orange = '#e4b81d'
    orange_pink = '#f0cca2'
    reddish_orange = '#d87733'
    red = '#bf1b4b'
    pink = '#f5dcd0'
    purplish_red = '#d14188'
    purplish_pink = '#f3d0db'
    red_purple = '#af2384'
    reddish_purple = '#c4408f'
    purple = '#f6559e'
    bluish_purple = '#5c66b1'
    purplish_blue = '#5879bf'
    blue = '#5c8aca'
    greenish_blue = '#6eafc7'
    bluegreen = '#5fa4be'
    bluish_green = '#18a279'


class HuePalette(AbstractEnum):
    color01 = '#e22c2c'
    color02 = '#e25a2c'
    color03 = '#e2872c'
    color04 = '#e2b52c'
    color05 = '#e2e22c'
    color06 = '#b5e22c'
    color07 = '#87e22c'
    color08 = '#5ae22c'
    color09 = '#2ce22c'
    color10 = '#2ce25a'
    color11 = '#2ce287'
    color12 = '#2ce2b5'
    color13 = '#2ce2e2'
    color14 = '#2cb5e2'
    color15 = '#2c87e2'
    color16 = '#2c5ae2'
    color17 = '#2c2ce2'
    color18 = '#5a2ce2'
    color19 = '#872ce2'
    color20 = '#8a2be2'
    color21 = '#b52ce2'
    color22 = '#e22ce2'
    color23 = '#e22cb5'
    color24 = '#e22c87'
    color25 = '#e22c5a'
    color26 = '#e22c2c'


def generate_pil(hexa: str) -> PIL.Image:
    rgb = RGB(hexa)
    return Image.new('RGB', (50, 50), rgb.rgb)


def save_pil(img: PIL.Image, path: Path):
    img.save(path, format='PNG')


def prepare_dirs(palette):
    # Generate colors
    for p_color in palette.values():
        pil_image = generate_pil(p_color)
        pil_path = COLORS_DIR_ASSIGNED / f'{p_color}.png'
        dir_path = COLORS_DIR_ASSIGNED / p_color
        os.makedirs(dir_path, exist_ok=True)
        save_pil(pil_image, pil_path)


def resolve_palette(palette, mod='RGB', n=1000):
    prepare_dirs(palette)

    for _ in range(0, n):
        if mod == 'RGB':
            color_instance = RGB.as_random_rgb()
        elif mod == 'LAB':
            color_instance = LAB.as_random_rgb()
        elif mod == 'NaiveRGB':
            color_instance = NaiveRGB.as_random_rgb()
        else:
            raise ValueError('')

        rgb_hex = color_instance.rgb_to_hex(color_instance.rgb)
        color_image = generate_pil(rgb_hex)
        assigned_color = color_instance.assing_to_palette(palette)

        if assigned_color is not None:
            color_path = COLORS_DIR_ASSIGNED / assigned_color / f'{rgb_hex}.png'
            save_pil(color_image, color_path)
        else:
            print(f'{rgb_hex} is too far from palette')


def get_random_rgb(n: int = 1000):
    return (np.random.random_sample(n * 3).reshape(n, 3) * 255).astype(int)


def show_random_lab_space():
    n_points = 10000
    data = (np.random.random_sample(n_points * 3).reshape(n_points, 3) * 255).astype(int)

    colors = [LAB(RGB.rgb_to_hex(c)) for c in data]
    srgb_colors = [c.rgb_decimal for c in colors]
    data = [(c.l, c.a, c.b) for c in colors]
    xs, ys, zs = list(zip(*data))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs, ys, zs, c=srgb_colors)
    plt.show()


def show_initial_centers_in_palette(palette):
    # Init plot.
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Centres.
    initial_centres = [LAB(color) for color in palette.values()]
    srgb_colors = [color.rgb_decimal for color in initial_centres]
    initial_centres_ = [(center.a, center.b, center.l) for center in initial_centres]
    xs, ys, zs = list(zip(*initial_centres_))

    # Plot centres.
    ax.scatter(xs, ys, zs, c=srgb_colors, s=100)

    # Generate color space.
    n_points = 10000
    data = (np.random.random_sample(n_points * 3).reshape(n_points, 3) * 255).astype(int)
    colors = [LAB(RGB.rgb_to_hex(c)) for c in data]
    srgb_colors = [c.rgb_decimal for c in colors]
    data = [(c.a, c.b, c.l) for c in colors]
    xs, ys, zs = list(zip(*data))
    ax.scatter(xs, ys, zs, c=srgb_colors, s=1)


def predict_palette(palette, save: bool = True):
    rgb_set = get_random_rgb(n=2000)
    lab_set_ = [LAB(RGB.rgb_to_hex(rgb)) for rgb in rgb_set]
    lab_set = [(lab.l, lab.a, lab.b) for lab in lab_set_]

    initial_centres_ = [LAB(color) for color in palette.values()]
    initial_centres = [(center.l, center.a, center.b) for center in initial_centres_]
    initial_colors = [color.rgb_decimal for color in initial_centres_]

    clusters, centres = clab_kmeans(lab_set, initial_centres)
    if save:
        # Save centres.
        np.save(str(COLORS_LAB_CENTRES), np.array(centres, dtype=np.float64))


def load_predicted_palette():
    class MockPalette(object):
        def __init__(self, hexa_colors):
            self.hexa_colors = ['#000000', '#ffffff', '#bcbebc'] + [_ for _ in hexa_colors]

        def values(self):
            return self.hexa_colors

    lab_centres = np.load(str(COLORS_LAB_CENTRES))
    hexa = [RGB.rgb_to_hex(LAB.lab_to_rgb(LabColor(*lab))) for lab in lab_centres]

    palette = MockPalette(hexa)
    return palette


def load_html_palette():
    class MockPalette(object):
        def __init__(self, hexa_colors):
            self.hexa_colors = [_ for _ in hexa_colors]

        def values(self):
            return self.hexa_colors

    df = pd.read_csv(HTML_COLORS)
    color_codes = df['Hex Color Code'].values
    return MockPalette(color_codes)


def main():
    # predict_palette(CIE2000Palette)
    # palette = load_predicted_palette()
    # palette = load_html_palette()
    palette = HuePalette
    resolve_palette(palette, 'LAB', n=5000)


if __name__ == '__main__':
    main()
