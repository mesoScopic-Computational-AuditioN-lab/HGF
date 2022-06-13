# HGF
Hierarchical Gaussian Filter (HGF) Toolbox

---

A python implimentation of the HGF model created by Christoph Mathys, 
current Python implimentation was fully recoded for Python by Jorie van Haren <jjg.vanharen@maastrichtuniversity.nl>.
Demo notebooks are made to give intuition of how to use the package.

Original refference:

Frässle, S., et al. (2021). TAPAS: An Open-Source Software Package for
Translational Neuromodeling and Computational Psychiatry. Frontiers in
Psychiatry, 12:680811. https://doi.org/10.3389/fpsyt.2021.680811

----

## Dependencies

| Required | Package           | Remarks         |
| ---------|-------------------|-----------------|
| Yes      | [Python 3]        |                 |
| Yes      | [numpy]           |                 |
| Yes      | [statsmodels.api] |                 |
| Yes      | [scipy]           | Opitimization   |
| No       | [pandas]          | Plotting        |
| No       | [seaborn]         | Plotting        |
| No       | [matplotlib]      | Plotting        |

----

## Installation

1. Clone the latest release and unzip it.
2. Change directory in your command line:
```
cd /path/to/HGF
```
3. Install HGF:
```
python setup.py install
```
4. Once the installation is complete, take a look at the demo notebook provided in `HGF Demo.ipynb`
