# Adversarial Defences with Manifold-Learning Flows

This repository contains code by Claire Bräuer, Pablo Robles Cervantes, Yufei Liu, and Chris Offner for a project that was done as part of the **Computational Intelligence Lab 2024** at ETH Zurich.

![](static/videos/readme_anim.gif)

Adversarial vulnerability remains a significant challenge for deep neural networks, as inputs manipulated with imperceptible perturbations can induce misclassification. Recent research posits that natural data occupies low-dimensional manifolds, while adversarial samples reside in the ambient space beyond these manifolds. Motivated by this _off-manifold hypothesis,_ we propose and examine a novel defense mechanism that employs **[manifold-learning normalizing flows ($\mathcal{M}$-Flows)](https://arxiv.org/abs/2003.13913)** to project input samples onto approximations of the data manifold prior to classification.

We illustrate the underlying principles of our method with a low-dimensional pedagogical example before testing its effectiveness on high-dimensional natural image data. While our method shows promise in principle on low-dimensional data, learning the data manifold proves highly unstable and sensitive to initial conditions. On image data, our method fails to surpass the baseline.

Supplementary animations that elucidate some of the discussed dynamics can be found on https://chrisoffner.github.io/mflow_defence/.

---

## Structure

### Jupyter notebooks

- `notebooks/two_spirals.ipynb`: Training the _Two Spirals_ classifier.
- `notebooks/attack_spiral_classifier.ipynb`: Adversarial attacks and defense of the _Two Spirals_ classifier. *Fig. 1** in the report was created here.
- `notebooks/defense_cases_frequency.ipynb`: Measuring the relative frequency of attack/defense cases **(A) - (D)** as described in **Sec. 3** of the report. **Fig. 5.** was created here.
- `notebooks/spiral_manifold_projection.ipynb`: Visualisations of the learned _on-manifold_ projection. **[Animations](https://chrisoffner.github.io/mflow_defence/)**, **Fig. 3**, and **Fig. 4** from the report were created here.
- ...

### Python scripts

- `notebooks/two_spirals_utils.py`: Generates the _Two Spirals_ dataset.
- `notebooks/generate_attacked_cifar10.py`: Generates dataset of adversarial FGSM and PGD attacks against the CIFAR-10 dataset.
- `notebooks/pixeldefend_cases_frequency.py`: Measures the relative frequency of attack/defense cases for the _PixelDefend_ baseline on CIFAR-10.
- ...