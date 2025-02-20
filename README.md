# Bollywood Movie Analysis

## Overview

This project analyzes Bollywood movie trailers, posters, and scripts to study gender representation using machine learning and computer vision techniques. The analysis includes:

- Emotion distribution in trailers
- Gender representation in movie posters
- Gender-based dialogue analysis in movie scripts

## Features

- **Emotion Analysis in Trailers:** Extracts frames from trailers, classifies emotions, and visualizes gender-specific emotional distribution.
- **Gender Classification in Movie Posters:** Uses a CNN model to classify male and female appearances in posters.
- **Dialogue Analysis in Scripts:** Extracts and swaps gender pronouns to analyze character dialogue distribution.

## Dataset

- **Trailer Data:** CSV file containing frame-wise gender and emotion annotations.
- **Movie Posters:** A dataset of Bollywood movie posters labeled with gender.
- **Scripts:** PDF movie scripts used for dialogue analysis.

## Project Structure

```
├── notebooks
│   ├── Data Visualiser
│   │   ├── data\_visual\_analysis.py
│   │   ├── identifier.py
│   ├── Image Gender Biaser
│   │   ├── biaser.py
│   │   ├── identifier.py
│   ├── solution_for_future_movies.py 
│   ├── wikipedia_analysis.ipynb 
├── README.md

```

on

### Prerequisites

Ensure you have Python 3.x installed along with the following dependencies:

```bash
pip install numpy pandas matplotlib opencv-python tensorflow scikit-learn PyPDF2 beautifulsoup4
```

## Usage

### 1. Visual Analysis

```bash
python notebooks/Data Visualiser/data_visual_analysis.py
```

Generates a pie chart of emotions categorized by gender.

### 2. Gender Classification in Movie Posters

```bash
python notebooks/Data Visualiser/identifier.py
```

Trains a CNN model and evaluates gender representation in movie posters.

### 3. Dialogue Analysis in Scripts

```bash
python analysis/script_dialogue_analysis.py
```

Extracts and swaps gender pronouns in dialogues for analysis.

## Results & Visualization

- **Pie chart** of emotional representation in trailers.
- **Bar chart** showing the count of male vs. female appearances in movie posters.
- **Processed scripts** with gender-swapped dialogues for analysis.

## Future Improvements

- **Expand Dataset:** Include more trailers, posters, and scripts.
- **Improve Model Accuracy:** Fine-tune CNN for better classification.
- **Add NLP Techniques:** Perform sentiment analysis on dialogues.



![image](https://github.com/user-attachments/assets/e0094668-a170-4887-860d-856d46a9faa4)
![image](https://github.com/yukeshwardevs/Fall_Intern_interview_project_ML_Q-3/assets/146966338/ee5f84fb-de51-43e6-a288-9934d7f86fb2)
![image](https://github.com/yukeshwardevs/Fall_Intern_interview_project_ML_Q-3/assets/146966338/d0dfb7fb-0cbc-4b29-8178-9626b393ee59)
![image](https://github.com/yukeshwardevs/Fall_Intern_interview_project_ML_Q-3/assets/146966338/a226242b-4e97-4827-8bd3-28d2e000e0c9)
![image](https://github.com/yukeshwardevs/Fall_Intern_interview_project_ML_Q-3/assets/146966338/8bd9ae11-bb71-413c-9557-102b35471154)
