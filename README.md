# Systems biology informed deep learning for inferring parameters and hidden dynamics

The code for the paper [A. Yazdani, L. Lu, M. Raissi, & G. E. Karniadakis. Systems biology informed deep learning for inferring parameters and hidden dynamics. *PLoS Computational Biology*, 16(11), e1007575, 2020](https://doi.org/10.1371/journal.pcbi.1007575).

## Code

The code depends on the deep learning package [DeepXDE](https://github.com/lululxvi/deepxde) v0.10.0. If you want to use the latest DeepXDE, you need to replace `dde.bc.PointSet` with `dde.PointSetBC`. You can also find the updated code at https://github.com/lu-group/sbinn.

- [glycolysis.py](glycolysis.py): Yeast glycolysis model
- [apoptosis.py](apoptosis.py): Cell apoptosis model
- [glucose_insulin.py](glucose_insulin.py): Ultradian endocrine model
- [FIM.ipynb](FIM.ipynb): Fisher information matrix. This code is written in Julia.

## Cite this work

If you use this code for academic research, you are encouraged to cite the following paper:

```
@article{yazdani2020systems,
  title   = {Systems biology informed deep learning for inferring parameters and hidden dynamics},
  author  = {Yazdani, Alireza and Lu, Lu and Raissi, Maziar and Karniadakis, George Em},
  journal = {PLoS computational biology},
  volume  = {16},
  number  = {11},
  pages   = {e1007575},
  year    = {2020}
}
```

## Questions

To get help on how to use the code, simply open an issue in the GitHub "Issues" section.
