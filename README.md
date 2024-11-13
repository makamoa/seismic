# WELCOME TO 


## License

Use of this software implies accepting all the terms and conditions described in
the license document available in this repository.  We remind users that the use of this
software is permitted for non-commercial applications, and proper credit must be
given to the authors whenever this software is used.

## Overall description

This project presents a novel algorithm designed to enhance the robustness of pre-trained deep learning models when analyzing real-time data contaminated by operational noise. By leveraging adversarial attacks, the algorithm embeds immunity against various noise profiles directly into the models, surpassing the limitations of standard noise augmentation techniques. This approach allows the models to focus on their primary tasks without being affected by complex noise types that are difficult to model, especially those with variable signal-to-noise ratios. The methodology is demonstrated on two tasks using a noisy seismic-while-drilling dataset: a simple first break picking task and a more challenging denoising task.

To validate the effectiveness of this technique, three deep learning models with different architectures (Unet, Restormer, and Swin Transformer) were trained on a clean synthetic dataset that simulates seismic-while-drilling data. These models were subjected to adversarial attacks to estimate the optimal noise that could influence their predictions. The performance of these adversarially trained models was compared to models trained using traditional noise augmentation methods. Results showed that models trained with adversarial attacks achieved higher accuracy and better generalizability, particularly in picking first breaks with an accuracy of 1 ms and providing detailed waveforms in denoising tasks. This technique reduces the need for manual noise modeling, making it a time-efficient and task-independent solution for enhancing model robustness against complex noise perturbations.

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
