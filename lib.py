# coding=utf-8

from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import math
from PIL import Image, ImageDraw, ImageFont, ImageChops
from skimage import exposure

def rescale_intensity(img, t_min=None, t_max=None):
  # Convert to RGB
  img = np.array(img)

  # Rescale
  img = img*1.0
  if t_min is None:
    t_min = np.min(img)

  if t_max is None:
    t_max = np.max(img)

  img = exposure.rescale_intensity(img, in_range=(t_min, t_max))
  img = (255*img).astype(np.uint8)

  img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

  return img

"""
Import image

Given a path, returns a PIL object. Optionally rescales the intensity
between min and max

Arguments:
  path str Path to the image
  t_min int Minimum value. Defaults to None
  t_max int Maximum value. Defaults to None

Output:
  PIL.Image
"""
def import_img(path, t_min=None, t_max=None):
  img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

  if img is None:
    raise Exception("No image found at " + str(path))

  # Convert to RGB
  img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

  # Rescale
  img = img*1.0
  if t_min is None:
    t_min = np.min(img)

  if t_max is None:
    t_max = np.max(img)

  img = exposure.rescale_intensity(img, in_range=(t_min, t_max))
  img = (255*img).astype(np.uint8)

  img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

  return img

def get_max_label_size(labels, font_size, font_path):
  font = ImageFont.truetype(str(font_path), size=font_size)
  font_height = 0
  font_width = 0
  for label in labels:
    if label == "":
      continue
    label = str(label).replace("\\n", "\n")
    img = Image.new('RGB', ( 1000, 1000 ))
    tmp_draw = ImageDraw.Draw(img)
    bbox = tmp_draw.textbbox((0,0), label, font=font, anchor='la')
    font_height = max(bbox[3], font_height)
    font_width = max(bbox[2], font_width)

  return (font_width, font_height)

def label_panel(panel, label, color, font_size, font_width, font_path, position='l', padding=20, y_offset=None):
  y_offset = padding if y_offset is None else y_offset
  font = ImageFont.truetype(str(font_path), size=font_size)
  label = str(label).replace("\\n", "\n")

  font_width += padding

  height = panel.size[1]
  width = panel.size[0] + font_width

  new = Image.new('RGB', (width, height), (255,255,255))

  if position == 'l':
    # image is on the right
    new.paste(panel, (font_width, 0))
  else:
    # image is on the left
    new.paste(panel, (0, 0))

  draw = ImageDraw.Draw(new)

  position_2_xy = {
    'l': (0,y_offset),
    'r': (panel.size[0]+padding, y_offset)
  }

  if position == 'l':
    # Right align text
    tmp = Image.new('RGB', ( 1000, 1000 ))
    tmp_draw = ImageDraw.Draw(tmp)
    bbox = tmp_draw.textbbox((0,0), label, font=font, anchor='la')
    this_width = bbox[2]

    if this_width < font_width:
      position_2_xy['l'] = ((font_width-this_width)-padding, y_offset)

  position_2_anchor = {
    'l': 'la',
    'r': 'ra'
  }

  position_2_align = {
    'l': 'right',
    'r': 'left'
  }

  if isinstance(color, str):
    # Get the color RGB values
    lut = get_lut(color, as_table=True)
    row = lut.iloc[200]
    color = (row['Red'], row['Green'], row['Blue'])

  color = "(" + ",".join(map(str,color)) + ")"

  draw.text(
    position_2_xy[position], 
    str(label), 
    fill='rgb' + color, 
    font=font,
    anchor=position_2_anchor[position],
    align=position_2_align[position]
  )

  return new


def get_blank_img(width, height):
  new = Image.new('RGB', (width, height), (255,255,255))

  return new

"""
Add a label to an image

Will add the label above or below the image.
The font color can be supplied as an (R,G,B) tuple or a string.
If a string, will attempt to load a LUT from ./luts/[color].lut
and will use the value corresponding for Index=200.

Arguments:
  img PIL.Image The image to label
  label str The label
  color string|tuple The font color
  font_size int The font size
  font_path Path The path to the font file
  position str 'tl', 'tr', 'bl', 'br', corresponding to top-right, top-left, bottom-left, bottom-right. Defaults to 'tl'
  label_padding int The amount of vertical padding in px between the text and the image. Defaults to 5

Output:
  PIL.Image
"""
def label_img(img, label, color, font_size, font_height, font_path, position='tl'):
  font = ImageFont.truetype(str(font_path), size=font_size)
  label = str(label).replace("\\n", "\n")

  position_2_anchor = {
    'tl': 'la',
    'tr': 'ra',
    'bl': 'ld',
    'br': 'rd'
  }

  padding = int(font_size/9)

  font_height += padding

  height = img.size[1] + font_height
  width = img.size[0]

  new = Image.new('RGB', (width, height), (255,255,255))

  if position == 'tl' or position == 'tr':
    # image is on the bottom
    new.paste(img, (0, font_height))
  else:
    # image is on the top
    new.paste(img, (0, 0))

  draw = ImageDraw.Draw(new)

  position_2_xy = {
    'tl': (0,0),
    'tr': (width, 0),
    'bl': (0, height),
    'br': (width, height)
  }

  if isinstance(color, str):
    # Get the color RGB values
    lut = get_lut(color, as_table=True)
    row = lut.iloc[200]
    color = (row['Red'], row['Green'], row['Blue'])

  color = "(" + ",".join(map(str,color)) + ")"

  draw.text(
    position_2_xy[position], 
    str(label), 
    fill='rgb' + color, 
    font=font,
    anchor=position_2_anchor[position]
  )

  return new


"""
Load a LUT

Will attempt to load a LUT from ./luts/[color].lut.

If as_table is False, will return a 1D array suitable for use by Image.point()

Arguments:
  color str The name of the LUT
  as_table bool Whether to return as a Pandas data frame or a list. Defaults to False

Output:
  (np.array|pd.DataFrame) The LUT
"""
def get_lut(color, as_table=False):
  color_path = Path(__file__ + "/../luts/" + color + ".lut").resolve()
  if color_path.exists() is not True or color_path.is_file() is not True:
    raise Exception("That LUT does not exist")
  lut = pd.read_csv(str(color_path))

  if as_table:
    return lut

  red = lut['Red']
  green = lut['Green']
  blue = lut['Blue']

  lut = np.concatenate((red, green, blue))

  return lut


"""
Merges BW images into a color-merged

Will merge black and white images using the supplied LUT names

Arguments:
  imgs list The list of PIL.Image objects to merge
  colors list The list of LUT names to use for each image

Output:
  PIL.Image
"""
def merge_imgs(imgs, colors):
  merged_img = Image.new('RGB', (imgs[0].size[0], imgs[0].size[1]), color=(0,0,0))
  for key, img in enumerate(imgs):
    lut = get_lut(colors[key])
    img = img.point(lut)
    merged_img = ImageChops.add(merged_img, img)

  return merged_img

def crop_img(img, factor, anchor='mm'):
  width = img.size[0]
  height = img.size[1]

  new_width = int(width / factor)
  new_height = int(height / factor)

  anchor_x = anchor[0]
  anchor_y = anchor[1]

  to_coords = (0,0)

  # Find the coords for cropping
  if anchor_x == 'm':
    from_x = int((width-new_width)/2)
  elif anchor_x == 'r':
    from_x = width-new_width
  else:
    from_x = 0

  if anchor_y == 'm':
    from_y = int((height-new_height)/2)
  elif anchor_y == 'b':
    from_y = height-new_height
  else:
    from_y = 0
  
  from_bb = (from_x, from_y, min(from_x+new_width, width), min(from_y+new_height, height))

  new = Image.new('RGB', (new_width, new_height))
  new.paste(img.crop(from_bb))

  return new

def draw_scale_bar(img, color, pixels_per_um, font_path=None, font_size=12, bar_width=20, bar_height=2, bar_padding=40, anchor='rb'):
  if font_path is not None:
    font = ImageFont.truetype(str(font_path), size=font_size)
    label = str(bar_width) + " µm"

  width = img.size[0]
  height = img.size[1]

  bar_width = int(bar_width*pixels_per_um)
  bar_height = int(bar_height*pixels_per_um) if height > 500 else 8
  bar_padding = bar_padding

  anchor_x = anchor[0]
  anchor_y = anchor[1]

  # Find the bounding box for the rectangle
  if anchor_x == 'm':
    x = int((width-bar_width-bar_padding)/2)
  elif anchor_x == 'r':
    x = width-bar_width-bar_padding
  else:
    x = 0+bar_padding

  if anchor_y == 'm':
    y = int((height-bar_height-bar_padding)/2)
  elif anchor_y == 'b':
    y = height-bar_height-bar_padding
  else:
    y = 0+bar_padding

  bb = [(x, y), (x+bar_width+1, y+bar_height+1)]

  if isinstance(color, str):
    # Get the color RGB values
    lut = get_lut(color, as_table=True)
    row = lut.iloc[200]
    color = (row['Red'], row['Green'], row['Blue'])

  color = "(" + ",".join(map(str,color)) + ")"

  draw = ImageDraw.Draw(img)
  draw.rectangle(bb, fill='rgb' + color)

  if font_path is not None:
    label_font_size = get_max_label_size([ label ], font_size, font_path)

    draw.text(
      ( int(x+bar_width/2-label_font_size[0]/2), y-label_font_size[1] ),
      label,
      fill='rgb' + color, 
      font=font,
      align='center'
    )

  return img

def assemble_panel(imgs, num_rows=1, margin=20, padding=20):
  num_imgs = len(imgs)
  img_width = imgs[0].size[0]
  if num_rows > 1:
    img_height = imgs[1].size[1]
    top_offset = imgs[0].size[1]-img_height
  else:
    img_height = imgs[0].size[1]
    top_offset = 0

  num_cols = math.ceil(num_imgs/num_rows)
  num_rows = np.min([ num_rows, math.ceil(num_imgs/num_cols) ])

  width =  num_cols*img_width+padding*(num_cols-1)+2*margin
  height = (top_offset+img_height)+(num_rows-1)*img_height+padding*(num_rows-1)+2*margin

  panel = Image.new('RGB', (width, height), (255,255,255))

  this_col = 0
  this_row = 0
  for idx,img in enumerate(imgs):
    x = margin+this_col*(img_width+padding)
    y = margin+this_row*(img_height+padding)
    if idx > 0:
      y += top_offset
    panel.paste(img, (x, y))

    this_col +=1
    if this_col == num_cols:
      this_col = 0
      this_row += 1

  return panel

