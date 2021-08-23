# Project sulcus

# Conda environment setup
conda create -n <environment_name>

conda install pytorch'=1.9.0' torchvision='0.10' cudatoolkit=10.2 -c pytorch
conda install scikit-learn openmpi -y
conda install  -c conda-forge opencv -y
pip install matplotlib visdom

conda install -c conda-forge torchio
conda install pandas
conda install -c daveeloo torchsummary
