## ACDC Light Sources dataset:

### Download:
You can download the ACDC light source panoptic annotations from [here](https://mega.nz/file/0jZniLSB#iVc0Qh894VvnMqM_Xj0sTv_M4E97CmdZ62QWIb2vIlM).

For more details on this dataset, check our [paper](https://ktzevel.github.io/SOLO/) and [supplementary material](https://ktzevel.github.io/SOLO/).


## Nighttime illuminants dataset:

### Download:
You can download our nighttime illuminants dataset [here](https://mega.nz/file/FqoSCABC#KuX3-Di-F6SSl9Y1Ji_bmjbDB3SeQSUUVJmHK_yyqks).


The dataset is a `json` file containing entries of the following structure:
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
- `name`: the filename of the image that this sample corresponds to.
- `cat`: the category that this samples belongs to. (e.g. 'street_light')
- `XYZ`: corresponds to the XYZ coordinates of each sample.
- `xy`: corresponds to the chromaticity coordinates of each sample.
- `XYZ_no_illum`: is `XYZ`  divided by `Y` and multiplied by `0.01`.
- `maxY`: is the maximum value of `Y` detected on the gray card.
- `argmaxY`: the coordinates of the aforementioned maximum value.
- `cct`: stands for the correlated color temperature of the sample.
- `iso`: is the `iso` setting of the camera while capturing the sample.
- `aperture`: is the `aperture` setting of the camera while capturing the sample.
- `focal_length`: is the `focal_length` setting of the camera while capturing the sample.

>These values assume as reference white the standard illuminant E of CIE.

### Other details:
- The camera used for the project is a `Canon 2000D`.
- The graycard used for the project is a `KODAK-Gray-Card-R27`.