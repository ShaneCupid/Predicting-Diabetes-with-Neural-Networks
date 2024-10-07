# Predicting Diabetes with Neural Networks

## Overview

This project was undertaken as part of the **AI 570 Deep Learning** course to predict the onset of diabetes in patients based on diagnostic measurements. Using the **Pima Indians Diabetes Dataset** from the UCI Machine Learning Repository, this project explores data preprocessing, neural network model building, and optimization using the Keras deep learning library.

## Technologies Used

- [Python 3](https://www.python.org/doc/) - Programming language used for data processing and model building.
- [Jupyter Notebook](https://jupyter-notebook.readthedocs.io/en/stable/) - An interactive environment for writing and testing code.
- [Pandas](https://pandas.pydata.org/docs/) - Library for data manipulation and analysis.
- [NumPy](https://numpy.org/doc/stable/) - Library for numerical computing in Python.
- [Matplotlib](https://matplotlib.org/stable/contents.html) - Library for visualizing data through plots.
- [Seaborn](https://seaborn.pydata.org/) - Data visualization library based on Matplotlib.
- [Scikit-learn](https://scikit-learn.org/stable/) - Library for machine learning, data preprocessing, and evaluation metrics.
- [TensorFlow & Keras](https://www.tensorflow.org/guide/keras) - Frameworks for building and training deep learning models.

## Description

The **Predicting Diabetes with Neural Networks** project focuses on developing, training, and evaluating neural networks to predict the onset of diabetes in patients. The dataset used in this project contains medical data such as the number of pregnancies, BMI, insulin levels, age, and more.

### Key Components of the Project:

- **Data Exploration & Preprocessing**:
  - Handled missing values, outliers, and duplicate records.
  - Performed data normalization and feature scaling.
  - Visualized data using histograms, boxplots, and pairplots to better understand feature relationships.
  
- **Baseline Neural Network**:
  - Built a simple feed-forward neural network using Keras Sequential API.
  - Compiled with **RMSProp optimizer**, binary cross-entropy loss, and accuracy as a metric.
  - Trained for 20 epochs with a batch size of 128.

- **Model Optimization**:
  - Built and trained additional models with varying configurations:
    1. Increased hidden layers and neurons.
    2. Added dropout layers to prevent overfitting.
    3. Experimented with different activation functions (e.g., ReLU, tanh).
    4. Tested different optimizers (e.g., RMSProp, Adam) and learning rates.

- **Model Evaluation**:
  - Evaluated models using metrics such as accuracy, loss, and **ROC-AUC**.
  - Compared the performance of multiple models to determine the best model for deployment.

### How It Works

1. **Data Loading**: The dataset is loaded into a Jupyter Notebook using Pandas for analysis and preprocessing.
2. **Data Preprocessing**:
   - Missing values for features such as glucose, blood pressure, and BMI are handled through imputation.
   - Outliers are detected and addressed, and the data is normalized using `StandardScaler`.
3. **Model Building**:
   - A **baseline neural network** is constructed with one hidden layer and ReLU activation.
   - Subsequent models explore different architectures and hyperparameters to improve performance.
4. **Model Training**:
   - The models are trained using training data, and performance is evaluated on a validation dataset.
   - The **best performing model** is saved for future use.
5. **Model Comparison**:
   - ROC curves are plotted for all models to visually compare performance, and AUC scores are calculated.

## Setup/Installation Requirements

To run this project, follow these steps:

1. Install Python 3 and the required packages:
    ```
    pip install jupyter pandas numpy matplotlib seaborn scikit-learn tensorflow
    ```
2. Clone the repository to your local machine:
    ```
    git clone https://github.com/yourusername/predicting-diabetes-neural-networks.git
    ```
3. Open the project directory and launch Jupyter Notebook:
    ```
    cd predicting-diabetes-neural-networks
    jupyter notebook
    ```
4. Open the notebook file `PredictingDiabetes.ipynb` to explore the project.

## Contact Information

For additional information or questions about this analysis, feel free to reach out:
- Business: [cupidconsultingllc@gmail.com](mailto:cupidconsultingllc@gmail.com)
- Personal: [shanecupid1@gmail.com](mailto:shanecupid1@gmail.com)
- Interested on how AI can automate your business ( I am currently working on applications in the Real Estate Investing Sector) ?:       [shane@maxpeakhomesolutions.com](malito:shane@maxpeakhomesolutions.com)
  https://maxpeakhomesolutions.com
  
## License

This project is licensed under the MIT License - see the LICENSE file for details.
