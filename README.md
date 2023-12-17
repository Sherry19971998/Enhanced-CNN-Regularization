# Enhanced-CNN-Regularization

## Overview

Convolutional neural networks (CNNs) possess the capability to learn powerful feature spaces. However, the complexity of tasks and model capacity mismatch makes CNNs susceptible to overfitting, necessitating proper regularization for effective generalization. In this project, we evaluate three advanced regularization methods—cutout, mixup, and self-supervised rotation prediction—applied to the ResNet-20 model on the CIFAR-10 dataset. The effectiveness of these methods is assessed under standard testing conditions, as well as against common image corruptions and white-box adversarial attacks (FGSM, rFGSM, random noise, and PGD). The study aims to determine the most effective strategy for enhancing the robustness and generalization ability of CNNs, providing valuable insights into tackling overfitting in complex neural network architectures.

## Table of Contents

1. [Overview](#overview)
2. [Models](#models)
3. [Installation](#installation)
4. [Contributors](#contributors)
5. [References](#references)
6. [Project Guidance](#project-guidance)
7. [Acknowledgements](#acknowledgements)



## Models

- **Step_1default.ipynb**
  - Initial training of ResNet-20 on CIFAR-10 with cross-entropy loss to establish a baseline model.
  - Hyperparameters: Initial learning rate of 0.1, weight decay of 1e-4, Cross-Entropy Loss, Stochastic Gradient Descent (SGD) with momentum of 0.9.
  - Training involves a learning rate decay at epochs 60, 120, and 160, spanning 200 epochs with a batch size of 128.
  - Best validation accuracy: 0.9145 with loss: 0.3077

- **Step_2cutout.ipynb**
  - Integration of the cutout regularizer with adjustments to hyperparameters (n_holes, length).
  - Testing with various parameters, recording best validation accuracy.
  - Best validation accuracy: 0.9285 with loss: 0.2441. The best parameter is n_holes=1, length= 20.
  - Increase accuracy by reducing overfitting as the network learns to recognize patterns with missing information. 
  - Results:

    | n_holes | length | Best Validation Accuracy | Loss    |
    |---------|--------|--------------------------|---------|
    | 1       | 8      | 0.9236                   | 0.3018  |
    | 1       | 12     | 0.9258                   | 0.2677  |
    | 1       | 16     | 0.9262                   | 0.2572  |
    | 1       | 20     | 0.9285                   | 0.2441  |
    | 1       | 24     | 0.9232                   | 0.2486  |
    | 2       | 8      | 0.9232                   | 0.2935  |
    | 2       | 12     | 0.9280                   | 0.2451  |
    | 2       | 16     | 0.9275                   | 0.2433  |
      
- **Step_3mixup.ipynb**
  - Implementation of the mixup regularizer with adjustments to the mix coefficient alpha.
  - Testing with various alpha values, recording best validation accuracy.
  - Best validation accuracy: 0.9252 with loss: 0.2650. The best parameter is alpha = 0.2.
  - It leads to a smoother decision boundary, which can also improve test accuracy by encouraging the model to generalize between classes. 
  - Results:

    | Alpha   | Best Validation Accuracy | Loss    |
    |---------|--------------------------|---------|
    | 0.1     | 0.9206                   | 0.2604  |
    | 0.15    | 0.9222                   | 0.2651  |
    | 0.2     | 0.9252                   | 0.2650  |
    | 0.4     | 0.9245                   | 0.3317  |
    | 1       | 0.9242                   | 0.3297  |

- **Step_4rotation.ipynb**
  - Implementation of the auxiliary rotation head on ResNet-20.
  - Training with the rotation head for self-supervision on CIFAR-10.
  - Best validation accuracy: 0.9092 with loss: 0.3341. The best parameter is lambda = 0.75.
  - Note: The Rotation method's underperformance stems from our deliberate choice to assess its standalone effectiveness without the PGD attack training used in the referenced paper. This approach aimed to isolate the impact of rotation-based self-supervision alone, revealing that, in this scenario, the method did not contribute significantly to improved classification and robustness.
  - Results:

    | Lambda  | Best Validation Accuracy | Loss    |
    |---------|--------------------------|---------|
    | 0.1     | 0.9003                   | 0.3652  |
    | 0.25    | 0.9013                   | 0.3551  |
    | 0.5     | 0.9044                   | 0.3216  |
    | 0.68    | 0.9066                   | 0.3115  |
    | 0.75    | 0.9092                   | 0.3341  |
    | 1       | 0.9023                   | 0.3278  |

- **Step_5all.ipynb**
  - Combination of all three methods with their best parameters.
  - The combination method has parameters nholes = 1 and length = 20 for the Cutout method, alpha = 0.2 for the Mixup method, and lambda = 0.75 for the Rotation method.
  - Best validation accuracy: 0.8918 with loss: 0.3261.
    
**Optional Requirements and Own Ideas**
- **Step_6test.ipynb**
  - Testing models on a corrupted variant of the CIFAR-10 test set with selected corruptions.
  - Rotation and Combination methods are least robust. Mixup consistently outperforms, while Cutout excels in brightness corruption but lacks stability compared to Mixup.
  - Results:


  | Corruption        | Gaussian Noise | Shot Noise | Impulse Noise | Brightness | Contrast |
  |-------------------|----------------|------------|---------------|------------|----------|
  | Original Resnet   | 0.299          | 0.2993     | 0.5814        | 0.9054     | 0.7881   |
  | Cutout            | 0.2132         | 0.2215     | 0.4695        | 0.9215     | 0.8214   |
  | Mixup             | 0.3647         | 0.3822     | 0.5938        | 0.918      | 0.8356   |
  | Rotation          | 0.1501         | 0.161      | 0.3579        | 0.9003     | 0.7804   |
  | Combination       | 0.1537         | 0.1648     | 0.3032        | 0.8878     | 0.7933   |
  
- **Step_4rotation_140epochs.ipynb**
  - New training method for rotation model with lambda=0.75 (100 epochs only for rotation loss and 40 epochs for both rotation and prediction loss).
  - Best validation accuracy: 0.8476 with loss: 0.4483.

- **Step_5all_140epochs.ipynb**
  - New training method for combine model with lambda=0.75 (100 epochs only for rotation loss and 40 epochs for both rotation and prediction loss).
  - Best validation accuracy: 0.8254 with loss: 0.5172.

Note：We trained Rotation for 200 regular epochs, mirroring other methods. Adapting the paper's approach (100 epochs with rotation loss only, then 40 epochs with classification loss) didn't yield substantial improvements. Nonetheless, it highlighted the effectiveness of rotation-based self-supervised training, showing notable enhancements within a short span of 40 epochs after incorporating classification loss.

- **attacker.py**
  - Testing the robustness of the best regularized models to white-box adversarial attacks including FGSM, rFGSM, random noise and PGD.
  - Epsilon values from 0 to 0.5 for each attack.
  - Mixup is the most robust method, while the Rotation method is comparatively weak.
  - Note: Rotation method's robustness relies on PGD attack in the original paper. Without PGD during training in our evaluation, Rotation method's standalone performance in basic classification is suboptimal, highlighting its limitations.

## Installation

Clone the repository:

```bash
# if using SSH
git clone git@github.com:jinws1999/Enhanced-CNN-Regularization.git

# if using HTTPS
git clone https://github.com/jinws1999/Enhanced-CNN-Regularization.git
```

## Contributors

- Weisheng Jin
- Chen Dong
- Xinyi Xie

## References

[1] DeVries, T., & Taylor, G. W. (2017). Improved regularization of convolutional neural networks with cutout. arXiv preprint arXiv:1708.04552.

[2] Hongyi Zhang, Moustapha Cissé, Yann N. Dauphin, David Lopez-Paz: mixup: Beyond Empirical Risk Minimization. ICLR (Poster) 2018

[3] Dan Hendrycks, Mantas Mazeika, Saurav Kadavath, Dawn Song: Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty. NeurIPS 2019

[4] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

[5] [Image Corruptions](https://github.com/bethgelab/imagecorruptions)

[6] Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572.

## Project Guidance

Special thanks to the Duke University ECE 661 Course Staff, including:

- TA Haoyu Dong
- Prof. Hai Li

We express our gratitude for their valuable guidance and support throughout the project.

## Acknowledgements

This project expresses gratitude to the referenced works for their valuable insights, which have significantly influenced and enriched our work. The project and its contributors do not claim ownership of the referenced works. The use of these references is solely for educational and research purposes, and the intellectual property rights remain with the respective authors and copyright holders. Any opinions, findings, conclusions, or recommendations expressed in this project are those of the authors and do not necessarily reflect the views of the original authors of the referenced works.
