# Code for illuminants dataset and nighttime image post-processing

## Contents:
- `camera_pipeline.m` and `camera_pipeline.py` (These files contain the main operations involved in a standard camera pipeline that takes as input a raw image and outputs a displayable sRGB image. We provide two equivalent versions, one for Python and one for Matlab.)
- `cct_utils.py` (Contains important utilities for color matrix manipulation and correlated color temperature).
- `generate_dataset.py` (A script that is used to create the nighttime illuminants dataset used in this work. In the first step the user has to annotate the part of the gray card that should be processed. Then the chromaticity of the sampled illuminant is calculated.)
- `select.py` (A utility that leverages matplotlib to provide an annotation interface used in `generate_dataset.py`)
- `post_processing.py` (That is the script used for the post-processing step of SOLO)