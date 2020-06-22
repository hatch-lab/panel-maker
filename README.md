# Panel maker

## Installation

```bash
git clone https://github.com/hatch-lab/panel-maker
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage
The basic usage pattern is to have a folder of individual TIFF files. This tool will combine those images into a single panel and, by default, generate a merged image. Images will be added left-to-right, top-to-bottom in alphabetical order.

### Basic usage
```bash
python make-panel.py /folder/of/tiff
```

Output are 3 files: `panel.tif`, `panel.jpg`, and `meta.json`. The first two are TIFF and JPEG versions of your panel, and the last is a text file containing information about how the panel was generated.

### Options

#### Levels
To set minimum and maximum intensity levels, use `--min` and `--max`. You may specify any integer 0–255. Each image in the panel can have its own levels: the 2nd image's levels are controlled by the 2nd instances of `--min` and `--max`.

`python make-panel.py /folder/of/tiff --min=10 --max=120 --max=250`
This will rescale the first image’s levels to be between 10 and 120 and the second image to be 0–250. No other images will be adjusted.

### Labels
The default labels for each image are simply numbers: 1, 2, … To specify your own labels, use `--label`. 

`python make-panel.py /folder/of/tiff --label="DAPI" --label="GFP" --label="TagRFP"`
This will set the first image’s label to DAPI, the second to GFP, and third to TagRFP.

### Colors
If your panel has a merged image (the default), you can specify the colors with `--hues`. The default hues are 180º, 58º, 300º. If you have more than three images, the rest will default to white.

The use of degrees corresponds to HSV color hue. You can use: http://hslpicker.com to find angles.

`python make-panel.py /folder/of/tiff --hues=150 --hues=300`
This will set the color of the first channel to green and the second to magenta.

If you want to use white as a color, this is specified by setting `--hues=-1`.

`python make-panel.py /folder/of/tiff --hues=150 --hues=300 --hues=-1`
This will set the color of the first channel to green, the second to magenta, and the third to white.

### Merged image
By default, all images are combined into a color merge, using the colors specified by `--hues`. If you don’t want to make a merged image, use `--skip-merge`.

`python make-panel.py /folder/of/tiff --skip-merge`
No merged image will be created.

### Rows
The default puts on images on one row. You may specify the number of rows with `--rows`.

```bash
python make-panel.py /folder/of/tiff --rows=2
```
This will arrange images onto 2 rows.

### Scale bar
By default, a 20 µm scale bar is added to the bottom right of an image. You can specify the width of the scale bar with `bar-microns`.

```bash
python make-panel.py /folder/of/tiff --bar-microns=10
```
This will add a 10 µm scale bar.

### Padding
By default, 10 px of white space are added between each image. You can change this with `--padding`.

```bash
python make-panel.py /folder/of/tiff --padding=0
```
Images will have no white space between them.

### Pixels per micron
By default, the how many pixels correspond to a micron are detected from the images metadata. However, this may not always be accurate. To specify the pixels/micron directly, use `--pixels-per-um`.

```bash
python make-panel.py /folder/of/tiff --pixels-per-micron=0.577
```
The scale bar will be adjusted assuming that 1 µm = 0.577 px.

## Help
If you need help, you can run:
```bash
python make-panel.py --help
```



