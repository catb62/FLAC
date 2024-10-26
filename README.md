# Federated Learning Defense Algorithm

This repository contains the implementation of a novel and efficient federated learning defense algorithm that leverages the Adam optimizer and incorporates both coarse-grained and fine-grained clustering mechanisms to neutralize backdoor attacks in a federated learning environment. The algorithm aims to accurately and efficiently identify malicious participants while mitigating the impact of non-stationary objectives and noisy gradients across multiple clients.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)

## Introduction

Federated learning is a distributed machine learning paradigm that enables multiple clients to collaboratively train a global model without sharing local data. This method is particularly vulnerable to backdoor attacks due to its distributed nature and inability to access local datasets. Our proposed algorithm addresses these challenges by using the Adam optimizer to mitigate noisy gradients and non-stationary objectives, along with employing clustering mechanisms to distinguish between benign clients and potential attackers.

## Features

- **Adam Optimizer**: Accelerates the learning process by mitigating the impact of noisy gradients and addressing non-stationary objectives.
- **Coarse-grained Clustering**: Uses a minimum spanning tree for initial clustering.
- **Fine-grained Clustering**: Further refines clustering to differentiate between benign clients and potential attackers.
- **Adaptive Clipping Strategy**: Alleviates the influence of malicious attackers.
- **Theoretical Analysis**: Demonstrates the consistent convergence of Adam in a federated backdoor defense environment.
- **Extensive Experiments**: Validates the effectiveness of the approach, showing superior performance compared to state-of-the-art baselines.

## Requirements

To run the code, ensure you have the following dependencies installed:

- Python 3.6+
- NumPy
- SciPy
- scikit-learn
- PyTorch
- NetworkX

You can install the required packages using pip:

```sh
pip install numpy scipy scikit-learn torch networkx
```

## Usage

To run the experiment, execute `tsadv.py` with optional command line parameters. For detailed parameter settings, please refer to the comments within `tsadv.py`.

```sh
python tsadv.py [options]
```

### Example

```sh
python tsadv.py
```

## Dataset

The default dataset used for testing is **FaceAll**. The script is configured to handle this dataset and can be modified to incorporate other datasets as needed.

## Results

The experimental results demonstrate that the proposed algorithm effectively identifies and mitigates the impact of backdoor attacks, outperforming state-of-the-art baseline methods.

---

For more information, please refer to the detailed documentation and comments within the codebase. If you encounter any issues or have questions, feel free to open an issue or contact the authors.