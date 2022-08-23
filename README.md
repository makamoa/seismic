# WELCOME TO 


## License

Use of this software implies accepting all the terms and conditions described in
the
[license](https://gitlab.kaust.edu.sa/makam0a/deepnano/-/blob/master/LICENSE)
document available in this repository.  We remind users that the use of this
software is permitted for non-commercial applications, and proper credit must be
given to the authors whenever this software is used.

## Overall description

This repository contains a demonstration of the flat optics design software ALFRED described in detail in the publication: 

*Broadband vectorial ultrathin optics with experimental efficiency up to 99% in the visible via universal approximators*

available as an open access article at [Light: Science & Applications volume 10, Article number: 47 (2021)](https://www.nature.com/articles/s41377-021-00489-7). 

The code makes use of the theory described in the publication:

*Generalized Maxwell projections for multi-mode network Photonics* [Scientific Reports volume 10, Article number: 9038 (2020)](https://doi.org/10.1038/s41598-020-65293-6)

Users are encouraged to read both publications and familiarize themselves with the underlying theory and logic behind the  software.

# Getting started

## Requierements

### Hardware

The codes provided are optimized for running on a CUDA capable NVIDIA GPU.
While not strictly required, the user is advised that the neural network training
process can take several hours when running on the GPU and may become prohibitively
long if running on a single CPU. 

### Software

The use of a Linux based operating system is strongly recommended. 
All codes were tested on a Ubuntu 18.04 system and Windows 10 system.

A working distribution of python 3.8 or higher is required.
The use of the [anaconda python distribution](https://www.anaconda.com/) is recommended
to ease the installation of the required python packages.

Examples of use of this software are provided as Jupyter notebooks and as such 
it requires the [Jupyter notebook](https://jupyter.org/) package. Note that this package
is included by default in the anaconda distribution.


## Initial set up

The usage of an Ubuntu 18.04 system or similar with a CUDA capable GPU and the anaconda python
distribution is assumed for the rest of this document. 

### System setup

The use of a separate python virtual environment is recommended for running the provided
programs. The file "deepnano.yml" is provided to quickly setup this environment in Linux
systems. To create an environment using the provided file and activate it do:

```bash
$ cd seismic
$ conda env create -f transformer.yml
$ conda activate transformer
```

To use a Jupyter notebook inside the created virtual environment, type the following code:

```bash
pip install ipykernel ipython kernel install --user --name=seismic
```
## Usage

Usage instructions are provided in the jupyter notebook files of the repository. Training examples are provided in runs/ directory
For validation on the real dataset please run EvalFieldDemo.ipynb

```bash
$ jupyter notebook EvalFieldDemo.ipynb
```
Please ensure the kernel is the correct one once the notebook starts running.
 
## Citing

When making use of the provided codes in this repository for your own work please ensure you reference the publication
