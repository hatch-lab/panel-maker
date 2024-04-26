# coding=utf-8

"""
Converts sets of TIFFs into a figure-ready panel

Usage:
  make-panel.py INPUT_DIR... 
  make-panel.py INPUT_DIR... [--out=<str>] [--zoom=1.0] [--zoom-anchor="mm"...] [--skip=1...] [--skip-except-merge=1...] [--gamma=1.0...] [--min=<int>...] [--max=<int>...] [--channel=<string>...] [--label=<string>...] [--color=<str>...] [--rows=1] [--pixels-per-um=5.58] [--bar-microns=20] [--merge-label=Merged] [--merge-mode=both] [--padding=10] [--bar-padding=2] [--label-font-size=55] [--bar-font-size=30] [--channel-font-size=40] [--title-font-size=40] [--n-channels=None] [--annotate-gradient=None] [--title=None] [--skip-bar] [--invert]

Arguments:
  INPUT_DIR  The directory with TIFF files of each channel. If multiple directories provided, a composite panel will be generated, saved to --out. Each needs to have the same channels and resolutions.

Options:
  -h --help  Show this screen.
  --min=<int>
  --max=<int>
  --gamma=<float>  [default: 1.0] Gamma correction function.
  --channel=<string>  Optional. The label for the channel.
  --label=<string>  Optional. The label of each set of images, or one label for all sets of images
  --color=<string>  The hues to use for the merged image. Defaults to yellow, magenta, turq, cyan.
  --rows=<int>  [default: 1] Number of rows we should have in our panel
  --pixels-per-um=<float>  Optional. Pixels per micron. Will attempt to extract this from the first TIFF found.
  --bar-microns=<int>  [default: 20] The width of the scale bar. Defaults to 20 um.
  --merge-label=<string>  [default: Merged] The label to use for the merged image
  --merge-mode=<string>  [default: both] If both, outputs grayscale channels and merge; skip, no merge; only, only the merge
  --padding=<int>  [default: 10] The number of pixels between each panel
  --bar-padding=<int>  [default: 20] The number of pixels to inset the scale bar
  --out=<str>  Required only if more than one INPUT_DIR is present. The output directory where the final TIFF is to be written.
  --zoom=<float>  [default: 1] If images should be zoomed in and cropped.
  --zoom-anchor=<str>  [default: mm] If zooming, whether to zoom in from the lt, lm, lr, mt, mm, mr, bt, bm, br (left-top, left-middle, left-right, middle-top, and so forth)
  --n-channels=<int> Instead of detecting the number of channels, specify it by hand
  --skip=<int> Channel to skip, indexed by 1
  --skip-except-merge=<int> Channel to skip except for the merge, indexed by 1
  --annotate-gradient=<str>  Add an ascending gradient to the left (if set to l) or right of all input panels
  --title=<str>  An overall title
  --label-font-size=<int>  The font size in points
  --channel-font-size=<int>  The font size in points
  --title-font-size=<int>  The font size in points
  --bar-font-size=<int>  The font size in points
  --skip-bar  Whether to skip the scale bar
  --invert  Invert the image

Output:
  A TIFF file
"""

import os
import re
from pathlib import Path
from docopt import docopt
from lib import import_img, label_img, merge_imgs, crop_img, assemble_panel, draw_scale_bar, get_max_label_size, label_panel, get_blank_img, rescale_intensity, add_gradient_triangle
import json
from schema import Schema, And, Or, Use, SchemaError, Optional
from tifffile import tifffile
import defusedxml.ElementTree as ET
import numpy as np
import cv2

def fill_list(val, min_len, default, end="end"):
  diff = min_len - len(val)
  if diff > 0:
    val = ([default]*diff + val) if end == "start" else (val + [default]*diff)

  return val

def get_tiff_paths(parent_path, skips):
  extensions = ["*.tif", "*.TIF", "*.tiff", "*.TIFF"]
  tiff_paths = []
  for e in extensions:
      tiff_paths.extend(list(parent_path.glob(e)))

  tiff_paths.sort(key=lambda x: str(x))
  tiff_paths = [ path for i,path in enumerate(tiff_paths) if path.name[0] != "." ]
  tiff_paths = [ path for i,path in enumerate(tiff_paths) if (i+1) not in skips ]
  if (parent_path / "params.json").exists() and (parent_path / "panel.tif").exists():
    tiff_paths = [ path for path in tiff_paths if path.name != "panel.tif" ]


  return tiff_paths

## Process inputs
arguments = docopt(__doc__, version='1.0')
schema_def = {
  'INPUT_DIR': [ os.path.exists ],
  '--out': Or(None, And(Use(os.path.expanduser), os.path.exists, error='--out does not exist')),
  '--min': [ Or(None, And(Use(int), lambda n: -1 <= n <= 255, error='--min must be greater than or equal to 0')) ],
  '--max': [ Or(None, And(Use(int), lambda n: -1 <= n <= 255, error='--max must be greater than or equal to 0')) ],
  '--gamma': [ Or(None, And(Use(float), lambda n: 0 <= n, error='--gamma cannot be negative')) ],
  '--channel': [ Or(None, lambda x: len(x) >= 0) ],
  '--label': [ Or(None, lambda x: len(x) >= 0) ],
  '--merge-label': lambda x: len(x) >= 0,
  '--title': Or(None, lambda x: len(x) >= 0),
  '--color': [ Or(None, lambda x: len(x) >= 0) ],
  '--rows': And(Use(int), lambda n: 0 < n, error='--rows must be an integer greater than or equal to 1'),
  '--pixels-per-um': Or(None, And(Use(float), lambda n: 0< n, error="pixels-per-um does not appear to be a number" )),
  '--bar-microns': And(Use(int), lambda n: 1 <= n, error="--bar-microns must be an integer greater than 0"),
  '--padding': And(Use(int), lambda n: 0 <= n, error="--padding must be greater or equal to 0"),
  '--bar-padding': And(Use(int), lambda n: 0 <= n, error="--bar-padding must be greater or equal to 0"),
  '--bar-font-size': Or(None, And(Use(int), lambda n: 1 < n, error="--bar-font-size must be greater than 1")),
  '--merge-mode': lambda x: x in ("both", "skip", "only"),
  Optional('--skip-bar'): bool,
  Optional('--invert'): bool,
  '--zoom': And(Use(float), lambda n: n >= 1, error="--zoom must be at least 1"),
  '--zoom-anchor': [ lambda n: n in [ 'lt', 'lm', 'lb', 'mt', 'mm', 'mb', 'rt', 'rm', 'rb' ] ],
  '--n-channels': Or(None, And(Use(int), lambda n: 0 < n, error='--n-channels must be greater than 0')),
  '--skip': [ Or(None, And(Use(int), lambda n: 0 < n, error="--skip must be greater than or equal to 1"))],
  '--skip-except-merge': [ Or(None, And(Use(int), lambda n: 0 < n, error="--skip-except-merge must be greater than or equal to 1"))],
  '--label-font-size': Or(None, And(Use(int), lambda n: 1 < n, error="--label-font-size must be greater than 1")),
  '--channel-font-size': Or(None, And(Use(int), lambda n: 1 < n, error="--channel-font-size must be greater than 1")),
  '--title-font-size': Or(None, And(Use(int), lambda n: 1 < n, error="--title-font-size must be greater than 1")),
  '--annotate-gradient': Or(None, lambda n: n in [ 'l', 'r' ], error='--annotated-gradient must be "l" or "r"')
}

schema = Schema(schema_def)

try:
  arguments = schema.validate(arguments)
except SchemaError as error:
  print(error)
  exit(1)

# Extra checking for --out and orthos
if len(arguments['INPUT_DIR']) > 1 and arguments['--out'] is None:
  print('If more than one directory is provided, an output directory must be provided using the --out option')
  exit(1)

img_dirs = [ Path(x).resolve() for x in arguments['INPUT_DIR'] ]
output_dir = img_dirs[0] if arguments['--out'] is None else Path(arguments['--out'])

skips = [ int(x) for x in arguments['--skip'] ]
skips_except_merge = [ int(x) for x in arguments['--skip-except-merge'] ]

# Get the # of channels
tiff_paths = get_tiff_paths(img_dirs[0], skips)
num_channels = len(tiff_paths)
if arguments['--n-channels'] is not None:
  num_channels = arguments['--n-channels']

if num_channels <= 0:
  print('No TIFFs found')
  exit(1)

# Get pixels_per_um
pixels_per_um = None
img_width = None
img_height = None
if arguments['--pixels-per-um'] is None:
  try:
    with tifffile.TiffFile(tiff_paths[0]) as tif:
      img = tif.pages[0].asarray()
      img_width = img.shape[1]
      img_height = img.shape[0]

      if 'spatial-calibration-x' in tif.pages[0].description:
        # Try from the description

        metadata = ET.fromstring(tif.pages[0].description)
        plane_data = metadata.find("PlaneInfo")

        for prop in plane_data.findall("prop"):
          if prop.get("id") == "spatial-calibration-x":
            pixels_per_um = 1/float(prop.get("value"))
            break
      
      elif 'XResolution' in tif.pages[0].tags:
        # Try from the XResolution tag
        pixels_per_um = tif.pages[0].tags['XResolution'].value

        if len(pixels_per_um) == 2:
          pixels_per_um = pixels_per_um[0]/pixels_per_um[1]
  except:
    pixels_per_um = float(input("Please enter pixels/µm:"))

else:
  with tifffile.TiffFile(tiff_paths[0]) as tif:
    img = tif.pages[0].asarray()
    img_width = img.shape[1]
    img_height = img.shape[0]
  pixels_per_um = float(arguments['--pixels-per-um'])

# Get color data
default_colors = [ "hatch-c4", "hatch-c3", "hatch-c2", "hatch-c1" ]
colors = [ re.sub("[^a-zA-Z0-9-]", "", color) for color in arguments['--color'] if Path("./luts/" + re.sub("[^a-zA-Z0-9-]", "", color) + ".lut").exists()  ]
if len(colors) < num_channels:
  colors.extend(default_colors[(4-num_channels-len(colors)):4])

# Get threshold params
min_thresholds = [ int(x) if x != -1 else None for x in arguments['--min'] ]
max_thresholds = [ int(x) if x != -1 else None for x in arguments['--max'] ]

min_thresholds = fill_list(min_thresholds, num_channels, None)
max_thresholds = fill_list(max_thresholds, num_channels, None)

# Label params
channel_labels = fill_list(arguments['--channel'], num_channels, "")
merge_label = arguments['--merge-label']
label_title = None
if len(arguments['--label']) == 1 and len(img_dirs) > 1:
  labels = fill_list([], len(img_dirs), "")
  label_title = arguments['--label'][0]
else:
  labels = fill_list(arguments['--label'], len(img_dirs), "")

if arguments['--label-font-size'] is not None:
  label_font_size = int(arguments['--label-font-size'])
else:
  label_font_size = int(70/1000*img_height*0.75)

if arguments['--channel-font-size'] is not None:
  channel_font_size = int(arguments['--channel-font-size'])
else:
  channel_font_size = int(70/1000*img_height*0.75)

if arguments['--bar-font-size'] is not None:
  bar_font_size = int(arguments['--bar-font-size'])
else:
  bar_font_size = int(50/1000*img_height*0.75)

if arguments['--title-font-size'] is not None:
  title_font_size = int(arguments['--title-font-size'])
else:
  title_font_size = int(100/1000*img_height*0.75)

# Get the rest of the arguments
num_rows = int(arguments['--rows'])
bar_microns = int(arguments['--bar-microns'])
panel_padding = int(arguments['--padding'])
bar_padding = int(arguments['--bar-padding'])
zoom = float(arguments['--zoom'])
zoom_anchor = fill_list(arguments['--zoom-anchor'], len(img_dirs), arguments['--zoom-anchor'][0])
skip_merge = False
skip_channels = False
if arguments['--merge-mode'] == "skip":
  skip_merge = True
elif arguments['--merge-mode'] == 'only':
  skip_channels = True
font_path = (Path(__file__).parent / "fonts/Geogrotesque-SemiBold.ttf").resolve()
add_triangle = arguments['--annotate-gradient']
title = arguments['--title']
gamma = [ float(x) for x in arguments['--gamma'] ]
gamma = fill_list(gamma, num_channels, 1.0)
invert = bool(arguments['--invert'])



if skip_merge:
  colors = [(0,0,0)] * 4


panels = []
label_font_width = get_max_label_size(labels, label_font_size, font_path)[0]
is_first = True
for input_key, img_dir in enumerate(img_dirs):
  label = labels[input_key]
  tiff_paths = get_tiff_paths(img_dir, skips)
  print(tiff_paths)
  imgs = []
  to_merge = []

  all_channel_labels = channel_labels.copy()
  all_channel_labels.append(merge_label)
  channel_font_height = get_max_label_size(all_channel_labels, channel_font_size, font_path)[1]
  
  for channel_key in range(0, num_channels):
    channel_label = channel_labels[channel_key]
    color = colors[channel_key]
    min_threshold = min_thresholds[channel_key]
    max_threshold = max_thresholds[channel_key]

    if len(tiff_paths) > channel_key and tiff_paths[channel_key].exists():
      img = import_img(tiff_paths[channel_key])
    else:
      img = get_blank_img(img_width, img_height)

    if zoom > 1:
      img = crop_img(img, zoom, zoom_anchor[input_key])
      
    img = rescale_intensity(img, min_threshold, max_threshold, gamma[channel_key], invert)

    if is_first:
      labelled_img = label_img(img, channel_label, color, channel_font_size, font_path, font_height=channel_font_height)
    else:
      labelled_img = label_img(img, "", color, channel_font_size, font_path, 0)
    
    if (channel_key+1) not in skips_except_merge and skip_channels is not True:
      imgs.append(labelled_img)
    to_merge.append(img)

  if not skip_merge:
    merged = merge_imgs(to_merge, colors, invert)
    if not skip_channels:
      if is_first:
        merged = label_img(merged, merge_label, (0,0,0), channel_font_size, font_path, font_height=channel_font_height)
      else:
        merged = label_img(merged, "", (0,0,0), channel_font_size, font_path, 0)
    imgs.append(merged)

  panel = assemble_panel(imgs, num_rows=num_rows, padding=panel_padding, margin=0)
  if len(label) > 0:
    if is_first:
      panel = label_panel(panel, label, (0,0,0), label_font_size, label_font_width, font_path, y_offset=channel_font_height+panel_padding)
    else:
      panel = label_panel(panel, label, (0,0,0), label_font_size, label_font_width, font_path)
  panels.append(panel)
  is_first = False

img = assemble_panel(panels, num_rows=len(panels), margin=panel_padding, padding=panel_padding)

if title is not None:
  img = label_img(img, title, (0,0,0), font_path=font_path, font_size=title_font_size)

# Add scale bar
if pixels_per_um is not None and not arguments['--skip-bar']:
  bar_color = (0,0,0) if invert else (255,255,255)
  img = draw_scale_bar(img, bar_color, pixels_per_um, zoom = zoom, font_size = bar_font_size, bar_width = bar_microns, bar_padding=(bar_padding+panel_padding), font_path=font_path)

if add_triangle is not None:
  triangle_offset = channel_font_height + get_max_label_size([title], title_font_size, font_path)[1]
  img = add_gradient_triangle(img, (0,0,0), position=add_triangle, height=channel_font_height, y_offset=triangle_offset, label=label_title, font_path=font_path, font_size=label_font_size, padding=panel_padding)

# Save
params = {
  "arguments": arguments,
  "min_thresholds": min_thresholds, 
  "max_thresholds": max_thresholds,
  "colors": colors,
  "bar_microns": bar_microns,
  "pixels_per_um": pixels_per_um,
  "merge_label": merge_label,
  "num_rows": num_rows,
  "skip_merge": skip_merge,
  "panel_padding": panel_padding,
  "label_font_size": label_font_size,
  "channel_font_size": channel_font_size,
  "zoom": zoom,
  "zoom_anchor": zoom_anchor
}

if pixels_per_um is not None:
  tiff_info = {
    282: params['pixels_per_um']/1E-6,
    283: params['pixels_per_um']/1E-6
  }
else:
  tiff_info = {}

img.save(output_dir / "panel.tif", tiffinfo=tiff_info)
img.save(output_dir / "panel.jpg")
with open(str((output_dir / "params.json")), 'w') as fp:
  fp.write(json.dumps(params))



