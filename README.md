# Dog Breed Classification & Localization using ML

## Machine Learning Pipeline

Developed a machine learning pipeline in Python using a subset of the Stanford Dogs dataset to perform two tasks:  

1. **Dog Breed Classification** (multi-class classification)  
2. **Dog Center Point Prediction** (regression)  

**Implemented and compared three core ML models:**  
- **Ridge Regression** for center point localization  
- **Logistic Regression with Softmax** for breed prediction  
- **K-Nearest Neighbors (K-NN)** for both tasks  

**Key contributions:**  
- Data preprocessing and input normalization  
- Feature engineering  
- Hyperparameter tuning  
- Exploration of different distance metrics and optimization strategies to improve performance and stability

Please refer to `report.pdf` for full details of our work.

## How to Run

Make sure the data is placed in the `dataset/` directory, that need to be created by extracting the dataset.zip file in its current location. And then run the code :

```bash
python main.py
```