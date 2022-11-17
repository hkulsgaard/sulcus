# Project sulcus

## Conda environment setup
  List of commands to install dependencies:
  
  ```
  conda create -n <env_name>
  conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
  conda install --channel conda-forge nibabel torchio opencv matplotlib visdom openmpi torchinfo
  conda install --channel anaconda scikit-learn pandas numpy
  ```
## Slurm configuration
  List of availeable GPU's for CUDA in our server:
  
- GM200 (GTX TITAN X)
- TU104 (RTX 2080 SUPER)
  
