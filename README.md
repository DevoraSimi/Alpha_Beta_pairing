
<p align="center">
  <img src="https://github.com/user-attachments/assets/b1d0e1bf-6b3d-4119-8902-f262ca67909d" alt="TCR-BARN Model" />
</p>
<p align="center">
    <a href="https://img.shields.io/badge/python-100%25-blue">
        <img alt="python" src="https://img.shields.io/badge/python-100%25-blue">
    </a>
    <a href="https://img.shields.io/badge/license-MIT-blue">
        <img alt="license" src="https://img.shields.io/badge/license-MIT-blue">
    </a>

**TCR-BARN** (TCR Beta-Alpha chains paiRing using Nlp) is a prediction model for TCRα and TCRβ chain binding, 
utilizing LSTM networks and one-hot encoding to capture sequence dependencies and gene usage. 


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Parameters](#parameters)
- [Examples](#Examples)
- [Structure](#structure)

## Installation
1. Download the repository.

2.  Install the required dependencies, run:
```bash
pip install -r requirements.txt
```
3. Navigate to the directory:
```bash
cd tcrbarn
```

## Usage
To run the prediction script, use the following command throw the command line:

```
python main.py --tcra <TCR_alpha_sequence> --va <V_alpha_gene> --ja <J_alpha_gene> --tcrb <TCR_beta_sequence> --vb <V_beta_gene> --jb <J_beta_gene> --data_type <data_type>
```

## Parameters
- [tcra]: TCR alpha sequence (string)
- [va]: V alpha gene (string, must start with "TRAV")
- [ja]: J alpha gene (string, must start with "TRAJ")
- [tcrb]: TCR beta sequence (string)
- [vb]: V beta gene (string, must start with "TRBV")
- [jb]: J beta gene (string, must start with "TRBJ")
- [data_type]: Type of data (string, optional, choices: 'All T cells', 'pMHC')

## Examples
To predict the binding probability for a given TCR alpha and beta sequence:
```
python main.py --tcra="CAVRDGGFGNVLHC" --va="TRAV03" --ja="TRAJ35" --tcrb="CASSYDNGGNTGELFF" --vb="TRBV06" --jb="TRBJ02-2" --data_type="pMHC"
```

## Structure

```
You can find the scripts and models in:
├───tcrbarn
│   ├───data
│   │   └───All T cells, pMHC1, pMHC2 .csv
│   │   └───process_data_new.py
│   ├───models
│   │   └───folder for each mode
│   │   │   └───best_alpha_encoder.pth
│   │   │   └───best_beta_encoder.pth
│   │   │   └───best_model.pth
│   ├───plots
│   │   └───.py files for plots
│   └───Loader.py
│   └───Models.py
│   └───Trainer.py
│   └───filtered_counters.json # For V J non rare genes.
│   └───hyperparameters.json 
│   └───main.py
│   └───v_j.py


```
