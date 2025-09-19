# Software Development Oriented to Machine Learning Project

Authors: Unai Lalana Morales and Eneko Isturitz Sesma

## Project Overview

SDOTML aims to bridge the gap between traditional software engineering and modern machine learning practices. It covers best practices, design patterns, and tools to build robust, maintainable, and scalable ML-driven applications.
In this project we will create a NN capable of predicting whether an image is AI generated or not.

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
├── notebooks/        
├── models/
├── reports/
├── my_project/
|   ├── dataset.py
|   └── train.py
├── README.md
├── pyproject.toml
└── uv.lock
```
Some folders will have .gitkeep placeholder files.
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