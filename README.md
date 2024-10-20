# CodonMPNN for Organism Specific and Codon Optimal Inverse Folding

### [Paper Link](https://arxiv.org/abs/2409.17265)

CodonMPNN is similar to ProteinMPNN but produces a codon sequences instead of an amino acid sequence and it is conditioned on a specific organism (the organism conditioning is optional).
The current version has been trained on Monomers from the AlphaFold database. 

### Conda environment
Here is one version of a conda environment that works for running the code. Keep in mind that the following assumes that you have a CUDA compatible GPU.
If that is not the case you should follow the pytorch installation instructions here instead: https://pytorch.org/get-started/locally/

```yaml
conda create  -n codon python=3.9
pip install jupyterlab
pip install numpy==1.21.2 pandas==1.5.3
pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install biopython==1.79 dm-tree==0.1.6 modelcif==0.7 ml-collections==0.1.0 scipy==1.7.1 absl-py einops
pip install pytorch_lightning==2.0.4 fair-esm mdtraj 
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@5484c38'
pip install biopython==1.81
pip install modelcif foldcomp tmtools  omegaconf rdkit-pypi imageio matplotlib plotly wandb torchdiffeq jupyterlab gpustat gemmi h5py deeptime 
```

# Training
A toy dataset is in `afdb_small` to run the code on. CodonMPNN was trained on a MMSeqs clustered version of AFDB.

[Here](https://publbuck.s3.us-east-2.amazonaws.com/afdb_with_groupings.csv) is a link to download the .csv file for the whole training validation and test data.
It contains AFDB ids for representatives of mmseqs clusters with a maximum 30% sequence identity. The train, val, test split is a random split.

Unfortunately, I cannot upload all the .pdb files somewhere due to the dataset size. You can download the pdb files from AFDB using their api and code similar to this:

```
import requests
url = f"https://alphafold.ebi.ac.uk/files/{afdb_id}.pdb"
response = requests.get(url)

if response.status_code == 200:
    with open(f'{afdb_id}.pdb', 'wb') as pdb_file:
        pdb_file.write(response.content)
```

# Inference
We provide the weights here.
```
https://publbuck.s3.us-east-2.amazonaws.com/codonmpnn_afdb_taxons20k.ckpt
```
