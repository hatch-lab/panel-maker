# coding=utf-8

"""
Converts a set of 3 or 4 TIFFs into a figure-ready panel

Usage:
  make-panel.py INPUT_DIR [--min=<int>...] [--max=<int>...] [--label=<string>...] [--color=<str>... | --colors=<str>] [--rows=1] [--pixels-per-um=5.58] [--bar-microns=20] [--merge-label=Merged] [--skip-merge] [--padding=10] [--bar-padding=20]

Arguments:
  INPUT_DIR  The path where the images are

Options:
  -h --help  Show this screen.
  --min=<int>  [default: 0]
  --max=<int>  [default: 255]
  --label=<string>  The label. Defaults to 1, 2, 3, ...
  --colors=<string>  If supplied, will be used for all panels; overrides hues.
  --color=<string>  [default: (255,0,255) (255,255,0) (0,255,255) (150,150,150)] The hues to use for the merged image. 
  --rows=<int>  [default: 1] Number of rows we should have in our panel
  --pixels-per-um=<float>  Pixels per micron.
  --bar-microns=<int>  [default: 20] The width of the scale bar. Defaults to 20 um.
  --merge-label=<string>  [default: Merged] The label to use for the merged image
  --skip-merge  Whether to skip a merge panel
  --padding=<int>  [default: 10] The number of pixels between each panel
  --bar-padding=<int>  [default: 20] The number of pixels to inset the scale bar from bottom right

Output:
  A TIFF file
"""

import sys
import os
from pathlib import Path
from docopt import docopt

import numpy as np
import cv2
import math
from ast import literal_eval

from PIL import Image, ImageDraw, ImageFont
from tifffile import tifffile

from skimage import exposure

import json

from schema import Schema, And, Or, Use, SchemaError, Optional

## Process inputs
arguments = docopt(__doc__, version='1.0')

schema = Schema({
  'INPUT_DIR': os.path.exists,
  '--min': [ And(Use(int), lambda n: 0 <= n, error='--min must be greater than or equal to 0') ],
  '--max': [ And(Use(int), lambda n: 0 <= n, error='--max must be greater than or equal to 0') ],
  '--label': lambda x: len(x) >= 0,
  '--merge-label': lambda x: len(x) >= 0,
  '--colors': Or(None, lambda x: len(x) >= 0),
  '--color': [ And(Use(literal_eval), lambda x: len(x) == 3 and 0 <= x[0] <= 255 and 0 <= x[1] <= 255 and 0 <= x[2] <= 255 ) ],
  '--rows': And(Use(int), lambda n: 0 < n, error='--rows must be an integer greater than or equal to 1'),
  '--pixels-per-um': Or(None, Use(float, error="pixels-per-um does not appear to be a number" )),
  '--bar-microns': And(Use(int), lambda n: 1 < n, error="--bar-microns must be an integer greater than 0"),
  '--padding': And(Use(int), lambda n: 0 <= n, error="--padding must be greater or equal to 0"),
  '--bar-padding': And(Use(int), lambda n: 0 <= n, error="--bar-padding must be greater or equal to 0"),
  Optional('--skip-merge'): bool
})

try:
  arguments = schema.validate(arguments)
except SchemaError as error:
  print(error)
  exit(1)

img_dir = Path(arguments['INPUT_DIR']).resolve()
output_dir = img_dir

tiff_paths = list(img_dir.glob("*.tif"))
tiff_paths.sort()

num_rows = int(arguments['--rows'])
pixels_per_micron = float(arguments['--pixels-per-um']) if arguments['--pixels-per-um'] else None
bar_microns = int(arguments['--bar-microns'])
skip_merge = bool(arguments['--skip-merge'])
panel_padding = int(arguments['--padding'])
bar_padding = int(arguments['--bar-padding'])

min_threshold = [ int(x) for x in arguments['--min'] ]
max_threshold = [ int(x) for x in arguments['--max'] ]

labels = arguments['--label'] if len(arguments['--label']) > 0 else list(range(len(tiff_paths)))

if arguments['--colors'] is not None:
  colors = [ literal_eval(arguments['--colors']) ]*len(tiff_paths)
else:
  colors = [ ( int(color[0]), int(color[1]), int(color[2]) ) for color in arguments['--color'] ]

def fill_list(val, min_len, default):
  diff = min_len - len(val)
  if diff > 0:
    val = val + [default]*diff

  return val

min_threshold = fill_list(min_threshold, len(tiff_paths), 0)
max_threshold = fill_list(max_threshold, len(tiff_paths), 255)
labels = fill_list(labels, len(tiff_paths), "")
colors = fill_list(colors, len(tiff_paths), ( 150, 150, 150 ))

merge_label = arguments['--merge-label']

font_path = Path("Roboto-Bold.ttf").resolve()


## Get images
images = [ cv2.imread(str(x), cv2.IMREAD_GRAYSCALE) for x in tiff_paths if x ]

if len(images) <= 0:
  print(arguments)
  exit()

# Get resolution
if pixels_per_micron is None:
  with tifffile.TiffFile(str(tiff_paths[0])) as img:
    pixels_per_micron = img.pages[0].tags['XResolution'].value
    if len(pixels_per_micron) == 2:
      pixels_per_micron = pixels_per_micron[0]
    dtype = img.pages[0].tags['XResolution'].dtype

    if dtype == '1I':
      # Convert from inches to microns
      pixels_per_micron = pixels_per_micron*3.937E-5
    elif dtype == '2I':
      # Convert from meters to microns
      pixels_per_micron = pixels_per_micron*1E-6
else:
  dtype = '2I'


## Convert everything to RGB so we're in a consisitent color space
images = [ cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) for x in images ]


## Rescale
for key in range(len(images)):
  px_min = min_threshold[key]
  px_max = max_threshold[key]
  img = images[key]*1.0
  images[key] = exposure.rescale_intensity(img, in_range=(px_min, px_max))


## Create merged image
if not skip_merge:
  merge_images = []

  for key in range(len(images)):
    img = images[key].copy()
    img[...,0] *= colors[key][0]
    img[...,1] *= colors[key][1]
    img[...,2] *= colors[key][2]

    merge_images.append(img.astype(np.uint8))
      
  # Blend
  merged = np.zeros(merge_images[0].shape, dtype=merge_images[0].dtype)
  for img in merge_images:
    merged = cv2.add(img, merged)

images = [ (255*img).astype(np.uint8) for img in images ]
images = images + [ merged ] if not skip_merge else images


## Generate panels
panel_width = images[0].shape[1]
panel_height = images[0].shape[0]

num_cols = math.ceil(len(images)/num_rows)
num_rows = np.min([ num_rows, math.ceil(len(images)/num_cols) ])

width = num_cols*panel_width+panel_padding*(num_cols+1)
height = num_rows*panel_height+panel_padding*(num_rows+1)

combined = np.zeros(( height, width, 3 )).astype(np.uint8)
combined[:] = 255
x = panel_padding
y = panel_padding

this_col = 0
this_row = 0
for img in images:
  combined[y:(panel_height+y), x:(panel_width+x)] = img
  this_col += 1

  if this_col == num_cols:
    this_col = 0
    x = panel_padding
    y += panel_height+panel_padding
  else:
    x += panel_width+panel_padding


## Add scale bar
bar_padding += panel_padding
bar_height = 10

bar_width = int(pixels_per_micron*bar_microns)

pt1 = (combined.shape[1]-bar_padding-bar_width, combined.shape[0]-bar_padding)
pt2 = (combined.shape[1]-bar_padding, combined.shape[0]-bar_padding-bar_height)
combined = cv2.rectangle(combined, pt1, pt2, (255,255,255), -1)


## Now add text, meta-data
combined = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(combined)

font = ImageFont.truetype(str(font_path), size=45)
x = panel_padding + 20
y = panel_padding + 20
labels = labels + [merge_label]

this_col = 0
this_row = 0
for label in labels:
  draw.text((x, y), str(label), fill='rgb(255,255,255)', font=font)
  this_col += 1

  if this_col == num_cols:
    this_col = 0
    x = panel_padding+20
    y += panel_height+panel_padding+20
  else:
    x += panel_width+panel_padding

tiff_info = {
  282: pixels_per_micron*1E-6,
  283: pixels_per_micron*1E-6,
  296: int(dtype[0])
}

combined.save(str(output_dir / "panel.tiff"), tiffinfo=tiff_info)
combined.save(str(output_dir / "panel.jpg"))

with open(str((output_dir / "meta.json")), "w") as fp:
  fp.write(json.dumps({ 
    "min_threshold": min_threshold, 
    "max_threshold": max_threshold,
    "bar_microns": bar_microns,
    "resolution": str(pixels_per_micron) + " px / um"
  }))
