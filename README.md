# CryoBench: Diverse and challenging datasets for the heterogeneity problem in cryo-EM

## Documentation

The latest documentation for CryoBench is available at our [homepage](https://cryobench.cs.princeton.edu/) and also at
our [manual](https://ez-lab.gitbook.io/cryobench).

For any feedback, questions, or bugs, please file a Github issue, start a Github discussion, or email.

## Installation
To run the metrics, you have to install `cryodrgn`.
We recommend installing `cryodrgn` in a clean virtual environment managed by [uv](https://docs.astral.sh/uv/)
(see the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it yet).

    # Create and activate a virtual environment
    $ uv venv .venvs/cryodrgn --python 3.9
    $ source .venvs/cryodrgn/bin/activate

    # install cryodrgn
    (cryodrgn) $ uv pip install cryodrgn

More installation instructions are found in the [documentation](https://ez-lab.gitbook.io/cryodrgn/installation).

Datasets are available for download at Zenodo.

1. Conf-het (IgG-1D, IgG-RL): [https://zenodo.org/records/11629428](https://zenodo.org/records/11629428).
2. Comp-het (Ribosembly, Tomotwin-100): [https://zenodo.org/records/12528292](https://zenodo.org/records/12528292).
3. Spike-MD: [https://zenodo.org/records/14941494](https://zenodo.org/records/14941494).

## Image Formation
Look at the repo [cryosim](https://github.com/ml-struct-bio/CryoBench/tree/main/cryosim).

## Metrics

### 1. Per-image FSCs
Look at the repo [metrics/fsc](https://github.com/ml-struct-bio/CryoBench/tree/main/metrics/fsc)

### 2. UMAP visualization
Look at the repo [metrics/visualization](https://github.com/ml-struct-bio/CryoBench/tree/main/metrics/visualization)

## Contact

Please submit any bug reports, feature requests, or general usage feedback as a github issue or discussion.

## Reference:

Jeon, Minkyu, et al. "CryoBench: Diverse and challenging datasets for the heterogeneity problem in cryo-EM." NeurIPS 2024 Spotlight. [paper](https://arxiv.org/abs/2408.05526).

```
@inproceedings{jeon2024cryobench,
 author = {Jeon, Minkyu and Raghu, Rishwanth and Astore, Miro and Woollard, Geoffrey and Feathers, Ryan and Kaz, Alkin and Hanson, Sonya M. and Cossio, Pilar and Zhong, Ellen D.},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {CryoBench: Diverse and challenging datasets for the heterogeneity problem in cryo-EM},
 year = {2024}
}
```
