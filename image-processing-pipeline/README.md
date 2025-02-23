# Code for illuminants dataset and nighttime image post-processing

## Contents:
- `camera_pipeline.m` and `camera_pipeline.py` (These files contain the main operations involved in a standard camera pipeline that takes as input a raw image and outputs a displayable sRGB image. We provide two equivalent versions, one for Python and one for Matlab.)
- `cct_utils.py` (Contains important utilities for color matrix manipulation and correlated color temperature).
- `generate_dataset.py` (A script that is used to create the nighttime illuminants dataset used in this work. In the first step the user has to annotate the part of the gray card that should be processed. Then the chromaticity of the sampled illuminant is calculated.)
- `select.py` (A utility that leverages matplotlib to provide an annotation interface used in `generate_dataset.py`)
- `post_processing.py` (That is the script used for the post-processing step of SOLO)


## Nighttime illuminants dataset:

### Download:
You can download our nighttime illuminants dataset [here](https://mega.nz/file/FqoSCABC#KuX3-Di-F6SSl9Y1Ji_bmjbDB3SeQSUUVJmHK_yyqks).


It is a `json` file containing entries of the following structure:
```json
{
        "name": "sample_name",
        "cat": "cat_name",
        "XYZ": [
            0.13183400342627444,
            0.14970064451537687,
            0.06429874156667122
        ],
        "xy": [
            0.3812066949744361,
            0.43286926322588154
        ],
        "XYZ_no_illum": [
            0.008806508739695693,
            0.01,
            0.004295154625073545
        ],
        "maxY": 0.23427327134751036,
        "argmaxY": [
            182,
            2446
        ],
        "cct": 4313,
        "iso": 200,
        "exposure_time": 0.5,
        "aperture": 5.6,
        "focal_length": 55
    },

```
### Properties:
- `XYZ` corresponds to the XYZ coordinates of each sample.
- `xy` corresponds to the chromaticity coordinates of each sample.
- `XYZ_no_illum` is `XYZ`  divided by `Y` and multiplied by `0.01`.
- `maxY` is the maximum value of `Y` detected on the gray card.
- `argmaxY` the coordinates of the aforementioned maximum value.
- `cct` stands for the correlated color temperature of the sample.
- `iso` is the `iso` setting of the camera while capturing the sample.
- `aperture` is the `aperture` setting of the camera while capturing the sample.
- `focal_length` is the `focal_length` setting of the camera while capturing the sample.

>These values assume as reference white the standard illuminant E of CIE.

### Other details:
- The camera used for the project is a `Canon 2000D`.
- The graycard used for the project is a `KODAK-Gray-Card-R27`.