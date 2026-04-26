import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# =====================
# 1. MA'LUMOT YUKLASH
# =====================
df = pd.read_csv('data/creditcard.csv')

print("=== DATASET MA'LUMOTLARI ===")
print(f"Jami tranzaksiyalar: {len(df):,}")
print(f"Firibgarlik (1): {df['Class'].sum():,}")
print(f"Oddiy (0): {(df['Class']==0).sum():,}")
print(f"Firibgarlik foizi: {df['Class'].mean()*100:.3f}%")

# =====================
# 2. TOZALASH
# =====================
df['NormalizedAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))
df = df.drop(['Time', 'Amount'], axis=1)

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE - kam klassni ko'paytirish
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"\nSMOTE dan keyin train size: {len(X_train_sm):,}")

# =====================
# 3. MODEL
# =====================
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_sm, y_train_sm)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

print("\n=== RANDOM FOREST NATIJASI ===")
print(classification_report(y_test, rf_pred, target_names=['Oddiy', 'Firibgarlik']))
print(f"ROC-AUC: {roc_auc_score(y_test, rf_prob):.4f}")

# =====================
# 4. GRAFIK
# =====================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Fraud Detection Natijalari', fontsize=14, fontweight='bold')

# Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Oddiy', 'Firibgarlik'],
            yticklabels=['Oddiy', 'Firibgarlik'], ax=axes[0])
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('Haqiqiy')
axes[0].set_xlabel('Bashorat')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, rf_prob)
axes[1].plot(fpr, tpr, color='red', label=f'ROC-AUC = {roc_auc_score(y_test, rf_prob):.4f}')
axes[1].plot([0,1],[0,1], 'k--')
axes[1].set_title('ROC Curve')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend()

plt.tight_layout()
plt.savefig('fraud_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nGrafik saqlandi: fraud_results.png")