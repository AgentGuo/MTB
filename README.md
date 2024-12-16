[![WWW](https://img.shields.io/badge/WWW'24-PASS-orange)](https://arxiv.org/pdf/2403.20150.pdf) [![license](https://img.shields.io/github/license/AgentGuo/MTB.svg)](https://img.shields.io/github/license/AgentGuo/MTB.svg) [![Visits Badge](https://badges.pufler.dev/visits/AgentGuo/MTB)](https://badges.pufler.dev/visits/AgentGuo/MTB)

# MTB: A Time-Series Prediction Benchmarking Tool Tailored to Enterprise Scenarios

## Introduction

MTB is a time-series prediction benchmarking
tool tailored to enterprise scenarios. MTB establishes prediction performance evaluation standards that align with enterprise expectations, enabling a fair comparison of time-series prediction algorithms based on their performance with real enterprise traffic data.

## Quickstart

1. **Install Dependencies**: Ensure you have Python and pip installed. Then, install the required dependencies using:
    ```bash
    pip install -r requirements.txt
    ```

2. **Load Data**: MTB implements the PyTorch dataset interface, and you can load data using the following code. For more details, refer to `example.py`:
    > The data will be automatically downloaded from Google Drive on the first run, or you can manually download it: [link](https://drive.google.com/drive/folders/1hn4jsjJQmZMAPJV3MKMi5tL5ey_9Spq5?usp=sharing).

    ```python
    train_dataset = mt_dataset.MTDataset(features='svc1', split="train")
    val_dataset = mt_dataset.MTDataset(features='svc1', split="val")
    test_dataset = mt_dataset.MTDataset(features='svc1', split="test")
    ```

## Cite

If you find this repo useful, please cite our paper.
```
@inproceedings{guo2024pass,
  title={PASS: Predictive Auto-Scaling System for Large-scale Enterprise Web Applications},
  author={Guo, Yunda and Ge, Jiake and Guo, Panfeng and Chai, Yunpeng and Li, Tao and Shi, Mengnan and Tu, Yang and Ouyang, Jian},
  booktitle={Proceedings of the ACM on Web Conference 2024},
  pages={2747--2758},
  year={2024}
}
```

## License

This dataset is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

You are free to:
- Share: Copy and redistribute the dataset in any medium or format.
- Adapt: Remix, transform, and build upon the dataset for any purpose, even commercially.

**Attribution Requirement**:  
If you use this dataset, please cite it as follows:

> Dataset Name, Author Name(s), Year. Licensed under CC BY 4.0. [Link to your repository or DOI].

For full license details, see [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).