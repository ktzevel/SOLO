# Sun Off, Lights On (SOLO): Photorealistic Monocular Nighttime Simulation for Robust Semantic Perception

__WACV 2025 (Oral)__

This repository represents the official implementation of the paper titled "Sun Off, Lights On: Photorealistic Monocular Nighttime Simulation for Robust Semantic Perception".

[![Website](docs/badges/badge-website.svg)](https://ktzevel.github.io/SOLO/)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](http://arxiv.org/abs/2407.20336)

[Konstantinos Tzevelekakis](https://scholar.google.com/citations?hl=de&user=8GEpNJYAAAAJ),
[Shutong Zhang](https://scholar.google.com/citations?user=JYMjWq8AAAAJ&hl=el&oi=sra),
[Luc Van Gool](https://scholar.google.com/citations?user=TwMib_QAAAAJ&hl=el&oi=sra),
[Christos Sakaridis](https://people.ee.ethz.ch/~csakarid/)

We present SOLO (Sun Off, Lights On), a monocular day-to-night translation method
that can be used to translate a daytime semantic segmentation dataset into a nighttime one
by converting each daytime image to its nighttime counterpart while preserving the original
semantic segmentation annotations. To the best of our knowledge, SOLO is the first monocular
physically based approach in 3D for simulating nighttime.

## 📢 News
2024-11-19: Initial setup, authors and citation related information is released.<br>
2025-02-23: Initial code uploaded.<br>


## 📌 To-Do
- [x] Upload project related information.
- [x] Upload code for SOLO pipeline.
- [x] Provide README files describing the functionality of the SOLO pipeline.
- [ ] Upload adapted [CISS](https://github.com/SysCV/CISS) `evaluation-code` for SOLO on Semantic Segmentation.	
- [ ] Provide instructions for running SOLO pipeline and results replication.

## 🎓 Citation

Please cite our paper:

```bibtex
@inproceedings{tzevelekakis2025solo,
author = {Tzevelekakis, Konstantinos and Zhang, Shutong and Van Gool, Luc and Sakaridis, Christos},
title = {{Sun Off, Lights On}: Photorealistic Monocular Nighttime Simulation for Robust Semantic Perception},
booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
year = {2025}}
```
## Acknowledgment
This work was supported by an ETH Career Seed Award.
