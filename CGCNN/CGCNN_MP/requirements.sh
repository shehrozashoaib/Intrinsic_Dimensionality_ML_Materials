conda create -n py312 python=3.12

# 1) activate your env
conda activate py312

# 2) add Jupyter kernel files for this env
python -m pip install --upgrade pip ipykernel
python -m ipykernel install --user --name py312 --display-name "Python 3.12 (py312)"

conda install -c conda-forge jarvis-tools

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install matminer pymatgen pandas numpy mp_api


git clone https://github.com/hackingmaterials/matbench
pip install --user ./matbench
