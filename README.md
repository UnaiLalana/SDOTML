# Software Development Oriented to Machine Learning Project

Authors: Unai Lalana Morales and Eneko Isturitz Sesma

## Project Overview

SDOTML aims to bridge the gap between traditional software engineering and modern machine learning practices. It covers best practices, design patterns, and tools to build robust, maintainable, and scalable ML-driven applications.
In this project we will create a NN capable of predicting whether an image is AI generated or not.


## License
This project (code and demo) is licensed under the [MIT License](./LICENSE).

## Project Demo
https://huggingface.co/spaces/EnekoIsturitz/SDOTML-demo

## Data Source


The dataset used for this project was sourced from Kaggle.

Dataset: AI vs. Human-Generated Images
Source: https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset
License: Apache License 2.0
Copyright: © 2025 alessandrasala79


We only used the train dataset as it was the only one labeled.

## Online Demo
You can try an online demo in:
[URL]

## Getting Started

1. **Clone the repository:**
    ```bash
    git clone https://github.com/UnaiLalana/SDOTML.git
    ```
2. **Install dependencies:**
    ```bash
    uv run sync #In the main folder
    ```

## Running the Training Script

To execute the performance analysis notebook using [uv](https://github.com/astral-sh/uv):

```bash
uv run --with jupyter jupyter lab
```

## Directory Structure

```
SDOTML/
├── data/        
│   ├── raw/
│   ├── interim/
|   |   ├── Initial_data/                   #Once the train is executed
|   |   ├── Initial_data.zip
|   |   └── Initial_data.csv
|   └── processed/
├── docs/
|   └── my_project/
|       ├── index.html
|       ├── datase.html
|       ├── net.html
|       ├── interactive_demo.html
|       └── train.html
├── notebooks/
|   ├── data_exploration.ipynb
|   └── performance_analysis.ipynb        
├── models/
├── reports/
|   ├── figures/
|   |   ├── Figure_1.png
|   |   ├── Figure_2.png
|   |   ├── Figure_3.png
|   |   ├── Figure_4.png
|   |   ├── Figure_5.png
|   |   ├── Figure_6.png
|   |   └── Figure_7.png
|   ├──Visualizaton_Report_SDOTML.pdf  
|   └──Visualizaton_Report_SDOTML.TEX    
├── my_project/
|   ├── __init__.py
|   ├── interactive_demo.py
|   ├── dataset.py
|   ├── net.py
|   └── train.py
├── README.md
├── .gitignore
├── pyproject.toml
└── uv.lock
```
Some folders might have .gitkeep placeholder files.
