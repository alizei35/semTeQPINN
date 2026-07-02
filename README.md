# Works on Trainbale Embedding Quantum Physics Informed Neural Networks
This project mainly focuses on doing benchmarks and developing the method TE-QPINN by Berger et al..

In this repository, the framework is implemented using **JAX**, **Optax** for L-BFGS optimization, and **PennyLane** for quantum circuit simulation.

Right now, only 1 example is available, and it is a highly oscillatory firsts order ODE over the domain $x \in [0,1]$:
$$\frac{du}{dx} = 4u - 6u^2 + \sin(50x) + u \cos(25x) - 0.5$$
Subject to the boundary condition:
$$u(0) = 0.75$$

To test this code, you can run `scripts/ode_optimization_optax.py`. To see the behavior of the method with the different number of qubits and layers, you can run `scripts/sweep.py`. The results and figures will be saved inside `results/` and `figures/` directory, respectively.
## References

If you use this work or codebase, please cite the original paper:

### APA Citation
Berger, S., Hosters, N. M., & Möller, M. (2025). Trainable embedding quantum physics informed neural networks for solving nonlinear PDEs. *Scientific Reports*, 15(1), 18823. [https://doi.org/10.1038/s41598-025-02959-z](https://doi.org/10.1038/s41598-025-02959-z)

### BibTeX
```bibtex
@article{berger2025trainable,
  title={Trainable embedding quantum physics informed neural networks for solving nonlinear PDEs},
  author={Berger, Stefan and Hosters, Norbert Michael and M{\"o}ller, Matthias},
  journal={Scientific Reports},
  volume={15},
  number={1},
  pages={18823},
  year={2025},
  publisher={Nature Publishing Group UK},
  doi={10.1038/s41598-025-02959-z}
}
