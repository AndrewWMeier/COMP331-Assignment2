# Gender Bias Mitigation in Income Prediction

## Project Overview
This project demonstrates a novel approach to mitigating gender bias in machine learning models using dynamic dataset augmentation. Utilizing the Adult Income dataset, the code implements a machine learning pipeline that:
- Loads and preprocesses income prediction data
- Detects and quantifies gender bias in the dataset
- Develops a dynamic augmentation strategy to reduce bias
- Trains and evaluates models with and without bias mitigation
- Generates visualizations to compare model fairness

## How To Run
1. Clone the repository
2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  #Activate on windows with `venv\Scripts\activate`
```

3. Install Dependencies
```bash
pip install -r requirements.txt 
```
4. Run 
```bash
python main.py 
```

## Example Results
![fairness_metrics](https://github.com/user-attachments/assets/988a69f1-465d-4366-aac2-4b3a567901cf)
