# Student Performance Prediction using Linear Regression

## Project Overview
This project predicts a student's final grade (`G3`) using the first-period grade (`G1`) and second-period grade (`G2`) with a Linear Regression model.

The project includes the original Jupyter Notebook analysis and an interactive Streamlit application.

## Objectives
- Explore the student performance dataset.
- Check missing values and duplicate records.
- Use `G1` and `G2` as input features and `G3` as the target.
- Train and evaluate a Linear Regression model.
- Compare actual and predicted final grades.
- Enter G1 and G2 values in the dashboard to estimate G3.

## Technologies
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Plotly
- Streamlit
- Jupyter Notebook

## Project Structure

```text
Student-Performance-Prediction-Linear-Regression/
├── app.py
├── Student_Performance.csv
├── Student_Performance_Prediction_Linear_Regression.ipynb
├── requirements.txt
└── README.md
```

## Run the Project

### 1. Open the project folder in VS Code or a terminal.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the dashboard

```bash
streamlit run app.py
```

The browser will open the Student Performance Prediction dashboard. Use the sidebar sliders to enter G1 and G2, then click **Predict Final Grade**.

## Model Details
- Algorithm: Linear Regression
- Input features: `G1`, `G2`
- Prediction target: `G3`
- Train/test split: 80% training and 20% testing
- Random state: 42
- Evaluation metrics: R² Score, Mean Absolute Error (MAE), Mean Squared Error (MSE)

The notebook removes duplicate rows before training. The Streamlit app applies the same duplicate-removal step and trains the model when the app starts.

## Dataset
`Student_Performance.csv` contains student-related information and grade columns. This project uses only `G1`, `G2`, and `G3` for the prediction workflow.

## Author
**Arya Gunji**

- GitHub: https://github.com/Arya-gunji
- LinkedIn: https://www.linkedin.com/in/arya-gunji/

## Disclaimer
This is an educational machine-learning project. Predicted grades are estimates and should not be used as the sole basis for academic decisions.
