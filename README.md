# CNC Milling Tool Wear Prediction

Using machine learning to predict tool condition (Healthy vs. Worn) in a CNC milling process based on sensor data. Looked at control errors, vibration, and spindle efficiency to identify the machine's degradation. This is a critical component of Predictive Maintenance in Industry 4.0 environments.

## How to Run
1. Clone the project: 
   ```bash
   git clone [https://github.com/ruciragati/cnc-milling-analysis-ml.git](https://github.com/ruciragati/cnc-milling-analysis-ml.git)
   cd cnc-milling-analysis-ml
2. Donwload data archives from [Kaggle](https://www.kaggle.com/datasets/shasun/tool-wear-detection-in-cnc-mill) and place it inside the folder.
3. Install & run: 
   ```bash
   pip install -r requirements.txt
   python cnc-milling.py
