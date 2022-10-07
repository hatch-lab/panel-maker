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

You 
can also stack multiple rows by providing multiple input folders and an output folder:
```bash
python make-panel.py /folder/of/tiff /folder/of/tiff2 --out=/combined
```

### Options

#### Levels
To set minimum and maximum intensity levels, use `--min` and `--max`. You may specify any integer 0–255. Each image in the panel can have its own levels: the 2nd image's levels are controlled by the 2nd instances of `--min` and `--max`.

```bash
python make-panel.py /folder/of/tiff --min=10 --max=120 --max=250
```
This will rescale the first image’s levels to be between 10 and 120 and the second image to be 0–250. No other images will be adjusted.

#### Channel labels
By default, channels will be unlabeled. To specify channel labels, use `--channel`:

```bash
python make-panel.py /folder/of/tiff --channel="TagRFP" --channel="GFP" --channel="DAPI"
```
This will set the first image’s label to DAPI, the second to GFP, and third to TagRFP.

#### Row labels
By default, each row of inputs will be unlabelled. But, if you have multiple conditions, it can be helpful to label them, using the `--label` option:

```bash
python make-panel.py /folder/of/tiff /folder/of/tiff2 --out=/combined --channel="TagRFP" --channel="GFP" --channel="DAPI" --label="Condition 1" --label="Condition 2"
```

#### Colors
If your panel has a merged image (the default), you can specify the colors with `--color`. This specifies which LUT to use, among those LUTs that are bundled with this program. By default, the last channel will be gray, the second from last will be cyan, the third from last will be magenta, and the fourth from last will be yellow. These were developed by Amanda. Alternatively, there are LUTs I took from: https://github.com/nelas/color-blind-luts, with the names:
* blue
* magenta
* orange
* red
* turq
* yellow

For example:

```bash
python make-panel.py /folder/of/tiff --channel="TagRFP" --channel="GFP" --channel="DAPI" --color=magenta --color=red --color=blue
```

#### Merged image
By default, all images are combined into a color merge, using the colors specified by `--color`. If you don’t want to make a merged image, use `--skip-merge`.

```bash
python make-panel.py /folder/of/tiff --skip-merge
```
No merged image will be created.

#### Scale bar
By default, a 20 µm scale bar is added to the bottom right of an image. You can specify the width of the scale bar with `bar-microns`.

```bash
python make-panel.py /folder/of/tiff --bar-microns=10
```
This will add a 10 µm scale bar.

#### Padding
By default, 10 px of white space are added between each image. You can change this with `--padding`.

```bash
python make-panel.py /folder/of/tiff --padding=0
```
Images will have no white space between them.

#### Zoom
If you would like to zoom in rather than showing the entire field, you can use the `--zoom` option. By default, it will zoom into the center of the field. You can change this with the `--zoom-anchor` option.

```bash
python make-panel.py /folder/of/tiff --channel="TagRFP" --channel="GFP" --channel="DAPI" --zoom=1.5 --zoom-anchor=rt`
```

`zoom-anchor` takes the following values:
* `lt`: The left-top of the field
* `mt`: Middle-top
* `rt`: Right-top
* `lm`: Left-middle
* `mm`: Center (the default)
* `rm`: Right-middle
* `lb`: Left-bottom
* `mb`: Middle-bottom
* `rb`: Right-bottom

If you have multiple inputs, you can specify multiple `zoom-anchor` values:

```bash
python make-panel.py /folder/of/tiff /folder/of/tiff2 --out=/combined --channel="TagRFP" --channel="GFP" --channel="DAPI" --zoom=1.5 --zoom-anchor=rt --zoom-anchor=lb
```

#### Pixels per micron
By default, the pixel to micron conversion ratio is extracted from the first TIFF’s metadata. However, this may not always be accurate. To specify the pixels/micron directly, use `--pixels-per-um`.

```bash
python make-panel.py /folder/of/tiff --pixels-per-micron=0.577
```
The scale bar will be adjusted assuming that 1 µm = 0.577 px.

## Help
If you need help, you can run:
```bash
python make-panel.py --help
```



