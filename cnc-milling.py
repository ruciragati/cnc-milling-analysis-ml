import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler

# load data
def load_data(path):
    main_df = pd.read_csv(f'{path}/train.csv')
    experiments = []
    
    for i in range(1, 19):
        exp = f"{i:02d}"
        file = f"{path}/experiment_{exp}.csv"

        if os.path.exists(file):
            df_exp = pd.read_csv(file)
            meta = main_df[main_df['No'] == i].iloc[0]

            df_exp['feedrate'] = meta['feedrate']
            df_exp['clamp_pressure'] = meta['clamp_pressure']
            df_exp['tool_condition'] = 1 if meta['tool_condition'] == 'worn' else 0
            
            experiments.append(df_exp)
    
    return pd.concat(experiments, ignore_index=True)

df = load_data('archive')

if not os.path.exists('data'):
    os.makedirs('data')

# preprocessing
active_processes = ['Layer 1 Up', 'Layer 1 Down', 'Layer 2 Up', 'Layer 2 Down', 'Layer 3 Up', 'Layer 3 Down']
df = df[df['Machining_Process'].isin(active_processes)].copy()

df['X1_ControlError'] = np.abs(df['X1_CommandPosition'] - df['X1_CurrentFeedback']) 
df['Y1_ControlError'] = np.abs(df['Y1_CommandPosition'] - df['Y1_CurrentFeedback'])
df['Z1_ControlError'] = np.abs(df['Z1_CommandPosition'] - df['Z1_CurrentFeedback'])
df['S1_Efficiency'] = df['S1_OutputPower'] / (df['S1_ActualVelocity'] + 1e-6)
df['S1_VibrationStd'] = df['S1_ActualAcceleration'].rolling(window=50).std()
df['S1_LoadMean'] = df['S1_OutputCurrent'].rolling(window=50).mean()
df.dropna(inplace=True)

features = [
    'X1_ControlError', 
    'Y1_ControlError', 
    'Z1_ControlError',
    'S1_Efficiency', 
    'S1_VibrationStd', 
    'S1_LoadMean',
    'S1_OutputPower',
    'S1_ActualVelocity',
    'feedrate', 
    'clamp_pressure'
]

corm = df[features].corr()

# heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corm, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Control Errors vs. Tool Condition')
plt.savefig('data/error_vs_condition.png', dpi=300, bbox_inches='tight')
plt.close()

# compare models
X = df[features]
y = df['tool_condition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM (SVC)": SVC(random_state=42),
    "Linear SVC": LinearSVC(max_iter=10000, random_state=42),
    "Perceptron": Perceptron(random_state=42),
    "SGD Classifier": SGDClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "Baseline (Dummy)": DummyClassifier(strategy="most_frequent")
}

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

results = {}

for name, model in models.items():
    try:
        model.fit(X_train_s, y_train)
        score = model.score(X_test_s, y_test)
        results[name] = score
    except Exception as e:
        print(f"Error -> {name}: {e}")

# results
leaderboard = pd.Series(results).sort_values(ascending=False)
leaderboard.index.name = 'Model'
leaderboard.to_csv('data/model_rank.csv', header=['Accuracy'])

# rank chart
plt.figure(figsize=(12, 8))
ax = sns.barplot(x=leaderboard.values, y=leaderboard.index, hue=leaderboard.index, palette='magma', legend=False)
for i, v in enumerate(leaderboard.values):
    ax.text(v + 0.01, i, f'{v:.4f}', color='black', va='center', fontweight='bold')

plt.title('Model Comparison')
plt.xlabel('Accuracy Score')
plt.ylabel('Model')
plt.xlim(0, 1.1) 
plt.tight_layout()
plt.savefig('data/model_rank_chart.png', dpi=300)
plt.close()

# best model
best_model_name = leaderboard.index[0]
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_s)

# confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Healthy', 'Worn'], 
            yticklabels=['Healthy', 'Worn'])
plt.title(f'Winner: {leaderboard.index[0]} Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('data/confusion_matrix.png', dpi=300)
plt.close()

# clamp vs vibration
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df.sample(2000), x='clamp_pressure', y='S1_VibrationStd', hue='tool_condition', alpha=0.5)
plt.title('Effect of Clamping Pressure on Vibration & Condition')
plt.xlabel('Clamp Pressure')
plt.ylabel('Vibration Standard Deviation')
plt.grid(True, alpha=0.3)
plt.savefig('data/clamp_vs_vibration.png', dpi=300, bbox_inches='tight')
plt.close()

# feature importance
if hasattr(best_model, 'feature_importances_'):
    plt.figure(figsize=(10, 6))
    importances = pd.Series(best_model.feature_importances_, index=features).sort_values()
    importances.plot(kind='barh', color='teal')
    plt.title('Most Critical Features')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('data/crit_feat.png', dpi=300)
    plt.close()

# validation
scores = cross_val_score(best_model, X_train.values, y_train.values, cv=10)

print("Best Model Stats :O")
print("Scores:", np.round(scores, 2))
print("Mean:", np.round(scores.mean(), 2))
print("Standard Deviation:", np.round(scores.std(), 2))

rf_model = RandomForestClassifier(n_estimators=100, oob_score = True)
rf_model.fit(X_train.values, y_train.values)

acc_rf_model = round(rf_model.score(X_train.values, y_train.values) * 100, 2)
print(f"Random Forest Training Accuracy: {round(acc_rf_model, 2)}%")

print("OOB Score:", round(rf_model.oob_score_, 4) * 100, "%")

# precision recall  
if hasattr(best_model, "predict_proba"):
    y_scores = best_model.predict_proba(X_train_s)[:, 1]
    precision, recall, threshold = precision_recall_curve(y_train, y_scores)

    def precision_recall(p, r, t):
        plt.figure(figsize=(14, 7))
        plt.plot(t, p[:-1], "r-", label="preecision", linewidth=5)
        plt.plot(t, r[:-1], "b-", label="recall", linewidth=5)

        plt.title('Precision vs. Recall Trade-off', fontsize=20)
        plt.xlabel("Threshold", fontsize=16)
        plt.ylabel("Score", fontsize=16)
        plt.legend(loc="lower left", fontsize=14)
        plt.ylim([0, 1.05])
        plt.grid(True, alpha=0.3)
        plt.savefig('data/precision_recall.png', dpi=300, bbox_inches='tight')
        plt.close()

precision_recall(precision, recall, threshold)