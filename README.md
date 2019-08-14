# Panel maker

## Installation

### Requirements
- macOS
- Xcode command line tools
- Homebrew
- Python 3.x
- virtualenv
- git
- GitHub account

### Installation of requirements
If you don’t have the above installed and are on macOS, follow the instructions below.

#### Xcode command line tools
1. Go to https://developer.apple.com/downloads/
2. Create an Apple account if you don’t have one; otherwise, login.
3. Find “Command Line Tools” in the list of downloads.
4. Download the installer.
5. Run the installer (this may take a long time).

#### Homebrew
Homebrew is software that makes it easier to install other programs (like Python).

1. Open Terminal.
2. Copy and paste the following, and hit Return:
  - `ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`
3. Copy and paste the following, and hit Return:
  - `echo 'PATH="/usr/local/opt/python/libexec/bin:$PATH"' >> ~/.bash_profile`

#### Python 3.7.x
1. In Terminal, copy and paste the following, and hit Return:
  - `brew install python`
2. Once that finishes, check to make sure you have Python 3.7.x installed by copy and pasting the following, then hitting Return:
  - `python --version`
3. You should see "Python 3.7.x" (where x is some number).
4. Finally, install virtualenv by copying and pasting the below, then hitting Return:
  - `pip install --user virtualenv`

#### git
1. Copy the line below and hit Return:
  - `brew install git`

#### GitHub
1. Signup for a free GitHub account at http://github.com.
2. Tell an administrator (right now that's Lucian) and they will set you up with the Hatch Lab GitHub repositories.

### Installation of the panel maker tool
1. Make or find a directory where you would like to install Hatch Lab tools
2. Everything will be placed inside of a `panel-maker` folder inside that directory.
3. In Terminal, type the following and hit Return:
  - `cd "~/path/to/my/folder"`
  - For example, if I want to install into `Documents/Hatch Lab`, I would type `cd "~/Documents/Hatch Lab"`
4. Retrieve the latest version of the tool by copying and pasting the following, then hitting Return:
  - `git clone --branch stable https://github.com/hatch-lab/panel-maker.git`
5. Copy and paste each line below and hit Return:
  - `cd panel-maker`
  - `virtualenv env`
  - `source ./env/bin/activate`
  - `pip install -r requirements.txt`
6. The tool should now be installed.

## Usage
The basic usage pattern is to have a folder of individual TIFF files. This tool will combine those images into a single panel and, by default, generate a merged image. Images will be added left-to-right, top-to-bottom in alphabetical order.

### Basic usage
`python make-panel.py /folder/of/tiff`

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
If your panel has a merged image (the default), you can specify the colors with `--hues`. The default hues are 180º, 58º, 300º, and 0º. If you have more than four images, the rest will default to 0º.

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

`python make-panel.py /folder/of/tiff --rows=2`
This will arrange images onto 2 rows.

### Scale bar
By default, a 20 µm scale bar is added to the bottom right of an image. You can specify the width of the scale bar with `bar-microns`.

`python make-panel.py /folder/of/tiff --bar-microns=10`
This will add a 10 µm scale bar.

### Padding
By default, 10 px of white space are added between each image. You can change this with `--padding`.

`python make-panel.py /folder/of/tiff --padding=0`
Images will have no white space between them.

### Pixels per micron
By default, the how many pixels correspond to a micron are detected from the images metadata. However, this may not always be accurate. To specify the pixels/micron directly, use `--pixels-per-um`.

`python make-panel.py /folder/of/tiff --pixels-per-micron=0.577`
The scale bar will be adjusted assuming that 1 µm = 0.577 px.

## Help
If you need help, you can run:
`python make-panel.py --help`



