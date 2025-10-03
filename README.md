# Software Development Oriented to Machine Learning Project

Authors: Unai Lalana Morales and Eneko Isturitz Sesma

## Project Overview

SDOTML aims to bridge the gap between traditional software engineering and modern machine learning practices. It covers best practices, design patterns, and tools to build robust, maintainable, and scalable ML-driven applications.
In this project we will create a NN capable of predicting whether an image is AI generated or not.

## Data Source

The dataset used for this project was sourced from Kaggle.
https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset
We only used the train dataset as it was the only one labeled.


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
|   |   └── Figure_6.png
|   └──Visualizaton_Report_SDOTML.pdf     
├── my_project/
|   ├── dataset.py
|   ├── net.py
|   └── train.py
├── README.md
├── pyproject.toml
└── uv.lock
```
Some folders might have .gitkeep placeholder files.
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

To execute the main training script using [uv](https://github.com/astral-sh/uv):

```bash
uv run src/train.py #In the my_project folder
```
