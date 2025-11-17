# Dynamic View Synthesis as an Inverse Problem [NeurIPS 2025]

[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://inverse-dvs.github.io/) [![arXiv](https://img.shields.io/badge/arXiv-2506.08004-b31b1b.svg)](https://arxiv.org/pdf/2506.08004)

<p align="center">

>[Hidir Yesiltepe](https://yesiltepe-hidir.github.io/), [Pinar Yanardag](https://pinguar.org/)

</p>

<p align="center">

</p>

<p align="center">
  <img src="assets/inverse_dvs.gif">
</p>

   From real-world complex scenes to AI-generated videos, our method preserves identity fidelity and synthesizes plausible novel views by operating entirely in noise initialization phase.

</p>

## Abstract

In this work, we address dynamic view synthesis from monocular videos as an inverse problem in a training-free setting. By redesigning the noise initialization phase of a pre-trained video diffusion model, we enable high-fidelity dynamic view synthesis without any weight updates or auxiliary modules. We begin by identifying a fundamental obstacle to deterministic inversion arising from zero-terminal signal-to-noise ratio (SNR) schedules and resolve it by introducing a novel noise representation, termed K-order Recursive Noise Representation. We derive a closed form expression for this representation, enabling precise and efficient alignment between the VAE-encoded and the DDIM inverted latents. To synthesize newly visible regions resulting from camera motion, we introduce Stochastic Latent Modulation, which performs visibility aware sampling over the latent space to complete occluded regions. Comprehensive experiments demonstrate that dynamic view synthesis can be effectively performed through structured latent manipulation in the noise initialization phase.

## Installation

This repository uses Python 3.10. We provide a setup script to create the virtual environment and install all dependencies.

```bash
bash setup_env.sh
```

This script will:
- Create a Python 3.10 virtual environment
- Install all required dependencies
- Fix any necessary compatibility issues
- Download required model checkpoints

## Usage

1. Adjust the configuration file `config.yml` to specify your settings:
   - GPU device IDs
   - Dataset paths and video files
   - Experiment trajectories
   - Inference parameters (inference steps, guidance scale, etc.)

2. Run the complete pipeline:

```bash
bash run_all.sh
```

The pipeline will execute the following steps:
1. **Preprocessing**: Extract depth information from input videos
2. **Camera Transformation**: Apply camera trajectories to generate warped views
3. **Inversion**: Invert the input video to latent noise representation
4. **Inference**: Generate novel views using the configured parameters

Results will be saved in the `results/` directory organized by trajectory and video name.

## Citation

If you find this work useful, please cite:

```bibtex
@article{yesiltepe2025dynamic,
  title={Dynamic View Synthesis as an Inverse Problem},
  author={Yesiltepe, Hidir and Yanardag, Pinar},
  journal={arXiv preprint arXiv:2506.08004},
  year={2025}
}
```