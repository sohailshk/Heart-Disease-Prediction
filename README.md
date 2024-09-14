# Heart Disease Prediction Model

## Overview
This project focuses on predicting the likelihood of heart disease using machine learning models. The dataset, obtained from [Kaggle](https://www.kaggle.com), contains over 900+ entries with 12 features related to patient health and medical history. By applying various data preprocessing techniques and machine learning algorithms, we aim to develop an accurate predictive model.

## Dataset
- **Source**: [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets)
- **Entries**: 900+
- **Features**: 20 (including age, cholesterol, blood pressure, chest pain type, etc.)
- **Target**: `HeartDisease` (binary classification: 0 for no disease, 1 for presence of disease)

## Project Workflow
1. **Data Preprocessing**:
   - Cleaned and handled missing values.
   - Categorical features (e.g., `Sex`, `ChestPainType`) were one-hot encoded.

2. **Feature Engineering**:
   - One-hot encoding of categorical variables such as `Sex`, `RestingECG`, `ExerciseAngina`, and `ST_Slope`.
   - Splitting the dataset into training (80%) and validation (20%) sets.

3. **Model Development**:
   - **Decision Tree Classifier**: Initially used to build the model, achieving an accuracy of **85.83% on the training set** and **86.41% on the validation set**.
   - **Random Forest Classifier**: Optimized the model by tuning hyperparameters and improving accuracy to **92.64% on training** and **89.13% on validation**.
   
4. **Evaluation**:
   - Tracked and plotted train and validation accuracies for different hyperparameters (e.g., `min_samples_split`).
   - Achieved optimal performance with **Random Forest Classifier**.

## Model Accuracy
| Model                  | Train Accuracy | Validation Accuracy |
|------------------------|----------------|---------------------|
| Decision Tree           | 85.83%         | 86.41%              |
| Random Forest           | 92.64%         | 89.13%              |

## Visualizations
The project includes plots comparing training and validation accuracy for different hyperparameter values such as `min_samples_split`.

![Train vs Validation Accuracy Plot](./accuracy_plot.png)

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - `pandas`: For data manipulation and preprocessing.
  - `scikit-learn`: For machine learning model development (Decision Tree, Random Forest).
  - `matplotlib`: For plotting and visualizations.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/sohailshk/Heart-Disease-Prediction.git
   ```
2. Install required libraries:
3.Run the notebook on google collab or Jupyter(I have Attached a google collab link above)

## Future Work
- Perform further hyperparameter tuning to improve validation accuracy.
- Integrate it with flask for people to get real time predictions instantly with more accurate model
- Will Deploy a report analyser for people that will help them make informed decisions early.
- Explore feature importance and its impact on predictions.

## Contributing
Feel free to fork this repository and submit pull requests if you'd like to contribute to the project.
