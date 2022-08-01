# coding=utf-8

"""
Converts a set of 3 or 4 TIFFs into a figure-ready panel

Usage:
  make-panel.py INPUT_DIR... [--out=<str>] [--ortho=<str>...] [--min=<int>...] [--max=<int>...] [--label=<string>...] [--input-label=<string>...] [--color=<str>... | --colors=<str>] [--rows=1] [--pixels-per-um=5.58] [--bar-microns=20] [--merge-label=Merged] [--skip-merge] [--padding=10] [--bar-padding=20] [--font-size=45] [--invert]

Arguments:
  INPUT_DIR  The directory with TIFF files of each channel. If multiple directories provided, a composite panel will be generated, saved to --out. Each needs to have the same channels and resolutions.

Options:
  -h --help  Show this screen.
  --min=<int>  [default: 0]
  --max=<int>  [default: 255]
  --label=<string>  The label. Defaults to 1, 2, 3, ...
  --input-label=<string>  Optional. Will be prepended to the first label of each row.
  --colors=<string>  If supplied, will be used for all panels; overrides hues.
  --color=<string>  [default: (255,0,255) (255,255,0) (0,255,255)] The hues to use for the merged image. 
  --rows=<int>  [default: 1] Number of rows we should have in our panel
  --pixels-per-um=<float>  Pixels per micron.
  --bar-microns=<int>  [default: 20] The width of the scale bar. Defaults to 20 um.
  --merge-label=<string>  [default: Merged] The label to use for the merged image
  --skip-merge  Whether to skip a merge panel
  --invert  Whether to invert the final image
  --padding=<int>  [default: 10] The number of pixels between each panel
  --bar-padding=<int>  [default: 20] The number of pixels to inset the scale bar from bottom right
  --font-size=<int>  [default: 45] The font size in points
  --ortho=<str>  Optional. If making an orthogonal projection, the x,y frames to center the projection on. Folders should be organized such that xy, xz, and yz folders are in the so-named directory for each panel.
  --out=<str>  Required only if more than one INPUT_DIR is present. The output directory where the final TIFF is to be written.

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

from skimage import exposure, util

import json

from schema import Schema, And, Or, Use, SchemaError, Optional

## Process inputs
arguments = docopt(__doc__, version='1.0')
schema_def = {
  'INPUT_DIR': [ os.path.exists ],
  '--out': Or(None, os.path.exists),
  '--min': [ And(Use(int), lambda n: 0 <= n, error='--min must be greater than or equal to 0') ],
  '--max': [ And(Use(int), lambda n: 0 <= n, error='--max must be greater than or equal to 0') ],
  '--label': lambda x: len(x) >= 0,
  '--input-label': [ Or(None, lambda x: len(x) >= 0) ],
  '--merge-label': lambda x: len(x) >= 0,
  '--colors': Or(None, lambda x: len(x) >= 0),
  '--color': [ And(Use(literal_eval), lambda x: len(x) == 3 and 0 <= x[0] <= 255 and 0 <= x[1] <= 255 and 0 <= x[2] <= 255 ) ],
  '--rows': And(Use(int), lambda n: 0 < n, error='--rows must be an integer greater than or equal to 1'),
  '--pixels-per-um': Or(None, And(Use(float), lambda n: 0< n, error="pixels-per-um does not appear to be a number" )),
  '--bar-microns': And(Use(int), lambda n: 1 <= n, error="--bar-microns must be an integer greater than 0"),
  '--padding': And(Use(int), lambda n: 0 <= n, error="--padding must be greater or equal to 0"),
  '--bar-padding': And(Use(int), lambda n: 0 <= n, error="--bar-padding must be greater or equal to 0"),
  '--font-size': And(Use(int), lambda n: 1 < n, error="--font-size must be greater than 1"),
  Optional('--skip-merge'): bool,
  Optional('--invert'): bool,
  '--ortho': [ Or(None, And(Use(literal_eval), lambda x: len(x) == 3 and 0 <= x[0] and 0 <= x[1] and 0 <= x[2])) ]
}

schema = Schema(schema_def)


  

def make_orthogonals(ortho, img_dir, padding, pixels_per_micron):
  x_idx = str(ortho[0])
  y_idx = str(ortho[1])
  z_idx = str(ortho[2])
  xy_paths = list((img_dir / "xy").glob("*_z*" + z_idx + "_c*.tif"))
  xz_paths = list((img_dir / "xz").glob("*_z*" + y_idx + "_c*.tif"))
  yz_paths = list((img_dir / "yz").glob("*_z*" + x_idx + "_c*.tif"))

  xy_paths.sort()
  xz_paths.sort()
  yz_paths.sort()

  ## Get images
  xy_images = [ cv2.imread(str(x), cv2.IMREAD_GRAYSCALE) for x in xy_paths if x ]
  xz_images = [ cv2.imread(str(x), cv2.IMREAD_GRAYSCALE) for x in xz_paths if x ]
  yz_images = [ cv2.imread(str(x), cv2.IMREAD_GRAYSCALE) for x in yz_paths if x ]

  if len(xy_images) <= 0 or len(xz_images) <= 0 or len(yz_images) <= 0:
    print("Couldn't find orthogonal slices in " + str(img_dir))
    exit(1)

  if len(xy_images) != len(xz_images) or len(xy_images) != len(yz_images):
    print("Length of xy, xz, and yz images are different")
    exit(1)

  # Get resolution
  if pixels_per_micron is None:
    with tifffile.TiffFile(str(xy_paths[0])) as img:
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

  tiff_info = {
    282: pixels_per_micron/1E-6,
    283: pixels_per_micron/1E-6,
    296: int(dtype[0])
  }

  ## Make images
  # Add xz to bottom of xy
  height = xy_images[0].shape[0] + padding + xz_images[0].shape[0]
  width = xy_images[0].shape[1] + padding + yz_images[0].shape[1]

  for key in range(len(xy_images)):
    combined = np.zeros(( height, width )).astype(np.uint8)
    combined[:] = 255

    # Draw a line
    gap_size = round(4*pixels_per_micron) # 5 um gap
    xy_images[key][0:ortho[1]-gap_size,ortho[0]] = 255
    xy_images[key][ortho[1]+gap_size:,ortho[0]] = 255
    xy_images[key][ortho[1],0:ortho[0]-gap_size] = 255
    xy_images[key][ortho[1],ortho[0]+gap_size:] = 255

    combined[0:xy_images[0].shape[0], 0:xy_images[0].shape[1]] = xy_images[key]
    combined[(xy_images[0].shape[0]+padding):height,0:xy_images[0].shape[1]] = xz_images[key]
    combined[0:xy_images[0].shape[0],(xy_images[0].shape[1]+padding):width] = yz_images[key]

    ## Now add text, meta-data
    combined = Image.fromarray(combined)
    file_name = str(key) + ".tif"

    combined.save(str(img_dir / file_name), tiffinfo=tiff_info)

  return xz_images[0].shape[0]

#
"""
Make a panel of images

Arguments:
  img_dir Path Path to the directory of channel TIFFs
  min_threshold list List of min values for each channel
  max_threshold list List of max values for each channel
  labels list List of labels to use for each channel
  merge_label string The label to use for the merged image
  colors list List of tuples giving the color to use in (R,G,B) format
  num_rows int The number of rows to use per panel
  pixels_per_micron float The pixels / µm
  skip_merge bool Whether to skip making a merge
  invert bool Whether to generate an inverted image
  panel_padding int The pixels of padding between each panel
  font_size int The font size to use

Output:
  PIL.Image The combined panel
"""
def make_panel(img_dir, min_threshold, max_threshold, labels, merge_label, colors, num_rows, pixels_per_micron, skip_merge, invert, panel_padding, font_size, **kwargs):
  tiff_paths = list(img_dir.glob("*.tif"))
  tiff_paths.sort()

  # Open images and convert everything to RGB so we're in a consisitent color space
  images = [ cv2.imread(str(x), cv2.IMREAD_GRAYSCALE) for x in tiff_paths if x ]
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
      # Our colors are R,G,B, but CV2 is B,G,R
      img[...,0] *= colors[key][2]
      img[...,1] *= colors[key][1]
      img[...,2] *= colors[key][0]

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


  ## Invert if necessary
  text_color = "(255,255,255)"
  if invert:
    combined = util.invert(combined)
    text_color = "(0,0,0)"

  ## Now add text, meta-data
  combined = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
  draw = ImageDraw.Draw(combined)

  font = ImageFont.truetype(str(font_path), size=font_size)
  x = panel_padding + 20
  y = panel_padding + 20
  labels = labels + [merge_label]

  this_col = 0
  for label in labels:
    label = str(label).replace("\\n", "\n")
    draw.text((x, y), str(label), fill='rgb' + text_color, font=font)
    this_col += 1

    if this_col == num_cols:
      this_col = 0
      x = panel_padding+20
      y += panel_height+panel_padding
    else:
      x += panel_width+panel_padding

  return combined

def fill_list(val, min_len, default, end="end"):
  diff = min_len - len(val)
  if diff > 0:
    val = ([default]*diff + val) if end == "start" else (val + [default]*diff)

  return val

def save_panel(output_dir, img, info):
  tiff_info = {
    282: info['pixels_per_micron']/1E-6,
    283: info['pixels_per_micron']/1E-6,
    296: int(info['dtype'][0])
  }

  combined.save(str(output_dir / "panel.tiff"), tiffinfo=tiff_info)
  combined.save(str(output_dir / "panel.jpg"))

  with open(str((output_dir / "meta.json")), "w") as fp:
    fp.write(json.dumps(info))

try:
  arguments = schema.validate(arguments)
except SchemaError as error:
  print(error)
  exit(1)

# Extra checking for --out and orthos
if len(arguments['INPUT_DIR']) > 1 and arguments['--out'] is None:
  print('If more than one directory is provided, an output directory must be provided using the --out option')
  exit(1)

if len(arguments['--ortho']) > 0 and len(arguments['--ortho']) != len(arguments['INPUT_DIR']):
  print('You must provide orthogonal coordinates for each set of channels.')
  exit(1)

img_dirs = [ Path(x).resolve() for x in arguments['INPUT_DIR'] ]
output_dir = img_dirs[0] if arguments['--out'] is None else Path(arguments['--out'])

# Get common params
num_rows = int(arguments['--rows'])
pixels_per_micron = float(arguments['--pixels-per-um']) if arguments['--pixels-per-um'] else None
bar_microns = int(arguments['--bar-microns'])
skip_merge = bool(arguments['--skip-merge'])
invert = bool(arguments['--invert'])
panel_padding = int(arguments['--padding'])
bar_padding = int(arguments['--bar-padding'])
font_size = int(arguments['--font-size'])

min_threshold = [ int(x) for x in arguments['--min'] ]
max_threshold = [ int(x) for x in arguments['--max'] ]

# Need the number of channels, pixels / um
tiff_paths = list(img_dirs[0].glob("*.tif"))
tiff_paths.sort()

labels = arguments['--label'] if len(arguments['--label']) > 0 else list(range(len(tiff_paths)))
input_labels = arguments['--input-label'] if len(arguments['--input-label']) > 0 else None

if arguments['--colors'] is not None:
  colors = [ literal_eval(arguments['--colors']) ]*len(tiff_paths)
else:
  colors = [ ( int(color[0]), int(color[1]), int(color[2]) ) for color in arguments['--color'] ]

min_threshold = fill_list(min_threshold, len(tiff_paths), 0)
max_threshold = fill_list(max_threshold, len(tiff_paths), 255)
labels = fill_list(labels, len(tiff_paths), "")
colors = fill_list(colors, len(tiff_paths), ( 150, 150, 150 ), "start")

merge_label = arguments['--merge-label']

font_path = Path("Roboto-Bold.ttf").resolve()

# Get pixels per microns
images = [ cv2.imread(str(x), cv2.IMREAD_GRAYSCALE) for x in tiff_paths if x ]

if len(images) <= 0:
  print("No images found! Searched in " + str(img_dirs[0]))
  exit(1)

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
    else:
      # Convert from meters to microns
      pixels_per_micron = pixels_per_micron*1E-6
      dtype = '2I'
else:
  dtype = '2I'


# Get panel-specific info
orthos = arguments['--ortho']

## Make panels
info = {
  "labels": labels,
  "input_labels": input_labels,
  "min_threshold": min_threshold, 
  "max_threshold": max_threshold,
  "colors": colors,
  "bar_microns": bar_microns,
  "pixels_per_micron": pixels_per_micron,
  "invert": invert,
  "dtype": dtype,
  "orthos": orthos,
  "merge_label": merge_label,
  "num_rows": num_rows,
  "skip_merge": skip_merge,
  "panel_padding": panel_padding,
  "font_size": font_size
}

panels = []

# First panel needs labels
if len(orthos) > 0:
  make_orthogonals(orthos[0], img_dirs[0], info)

_labels = info['labels']
_merge_label = info['merge_label']
if input_labels is not None and len(input_labels) >= 1:
  _labels = [ "\n" + s for s in info['labels'] ]
  _labels[0] = info['input_labels'][0] + "\n" + info['labels'][0]
  _merge_label = "\n" + _merge_label
panels.append(make_panel(img_dirs[0], **{**info, 'labels': _labels, 'merge_label': _merge_label}))

extra_bar_padding = 0
for idx,img_dir in enumerate(img_dirs[1:]):
  key = idx+1 # The index gets rekeyed when we slice
  if len(orthos) > 0:
   extra_bar_padding = make_orthogonals(orthos[key], img_dirs[key], info)

  # Don't print labels on subsequent panels
  _labels = []
  if len(info['input_labels']) > key:
    _labels = [ "" ] * len(info['labels'])
    _labels[0] = info['input_labels'][key]
  panels.append(make_panel(img_dirs[key], **{**info, 'labels':_labels, 'merge_label':''}))


# Now assemble the panels
# Convert from PIL back into numpy
combined = np.array(panels[0])
for panel in panels[1:]:
  np_panel = np.array(panel)

  border_size = abs(np_panel.shape[1]-combined.shape[1])
  if combined.shape[1] > np_panel.shape[1]:
    # This panel is narrower; we add a border
    border_size = ( np_panel.shape[0], border_size, 3 )
    border = np.ones(border_size, dtype=np_panel.dtype)
    border *= 0 if invert else 255
    np_panel = np.concatenate((np_panel, border), axis=1)
  elif np_panel.shape[1] > combined.shape[1]:
    # This panel is wider; we add a border to combined
    border_size = ( combined.shape[0], border_size, 3 )
    border = np.ones(border_size, dtype=combined.dtype)
    border *= 0 if invert else 255
    combined = np.concatenate((combined, border), axis=1)

  # Remove the bottom border
  combined = combined[0:combined.shape[0]-panel_padding,:]

  combined = np.concatenate((combined, np_panel), axis=0)

## Add scale bar
bar_color = (0,0,0) if invert else (255,255,255)

bar_padding += panel_padding
bar_height = 10

bar_width = int(pixels_per_micron*bar_microns)

pt1 = (combined.shape[1]-bar_padding-bar_width, combined.shape[0]-bar_padding)
pt2 = (combined.shape[1]-bar_padding, combined.shape[0]-bar_padding-bar_height)
combined = cv2.rectangle(combined, pt1, pt2, bar_color, -1)

# Now get us back to PIL
combined = Image.fromarray(combined)

save_panel(output_dir, combined, info)
