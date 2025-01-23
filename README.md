
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
To install the required dependencies, run:
```bash
pip install -r requirements.txt
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
- [data_type]: Type of data (string, choices: 'All T cells', 'pMHC-I')

## Examples
To predict the binding probability for a given TCR alpha and beta sequence:
```
python main.py --tcra="CAVRPQNYGQNFVF" --va="TRAV3" --ja="TRAJ26" --tcrb="CSVVVTLDEQFF" --vb="TRBV29-1" --jb="TRBJ2-1" --data_type="All T cells"
```

## Structure

```

You can find the scripts and models in:
├───tcrbarn
│   ├───models
│   │   └───alpha_encoder_irec.pth
│   │   └───alpha_encoder_vdjdb.pth
│   │   └───beta_encoder_irec.pth
│   │   └───beta_encoder_vdjdb.pth
│   │   └───model_irec.pth
│   │   └───model_vdjdb.pth
│   └───Loader.py
│   └───Models.py
│   └───Trainer.py
│   └───best_hyperparameters_ireceptor.json # For All T cells model.
│   └───best_hyperparameters_vdjdb.json # For pMHC-I model.
│   └───filtered_counters.json # For V J non rare genes.
│   └───main.py
│   └───v_j.py


```
