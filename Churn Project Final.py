# %%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("Customer-Churn-Records.csv")




# %%
df.isnull().sum()


# %%
plt.rcParams['font.family'] = 'DejaVu Sans'
df['Exited'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Customer Retention vs Churn')
plt.xlabel('Exited (0 = Retained, 1 = Churned)')
plt.ylabel('Number of Customers')
plt.show()


plt.hist(df['Age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Customer Age Distribution')
plt.xlabel('Age')
plt.ylabel('Number of Customers')
plt.show()

# %%
df.head()

# %%
df = pd.read_csv("Customer-Churn-Records.csv")
num_df = df.select_dtypes(include=[np.number])  # ตัด Geography, Gender, Surname ฯลฯ ออก

# 1) คำนวณค่าสหสัมพันธ์
corr = num_df.corr(method='pearson')


# 2) วาด heatmap
plt.figure(figsize=(12, 9))
sns.heatmap(
    corr, 
    cmap='YlOrBr',       # โทนเหลือง-ส้ม-น้ำตาลเหมือนตัวอย่าง
    annot=True,          # ใส่ตัวเลขในช่อง
    fmt=".2f",           # แสดงทศนิยม 2 ตำแหน่ง
    vmin=-1, vmax=1,     # สเกลเต็มช่วง -1..1
    linewidths=.5,       # เส้นแบ่งช่อง
    cbar_kws={'shrink': .8}
)
plt.title("Correlation Heatmap (Numeric Features)", pad=12)
plt.tight_layout()
plt.show()

# %%
#เลือกใช้ข้อมูลตามนี้จากการเช็ค กราฟความสัมพันธ์การ เทรนโมเดลข้างบน
cols = ["Complain", "Age", "NumOfProducts", "IsActiveMember", "Balance", "CreditScore", "Tenure","HasCrCard","Exited"]
df = df[cols]
df


# %%
df.Age.value_counts()


# %%
#ไล่เช็คข้อมูลของ Age
df.Age.hist(bins=20)


# %%
#เช็คค่าของข้อมูลที่จะใช้ในการเทรน
df.describe()


# %%
#เริ้มเทรนโมเดลโดย Random Forest 
X = df[
    ['Complain',
      'Age',
      'NumOfProducts',
      'IsActiveMember',
      'Balance',
      'CreditScore',
      'Tenure',
      'HasCrCard']
      ]
Y = df['Exited']
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

X_train.shape


# %%
#จูน Model 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
rfc=RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100,200,300,400,500,600],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [2,3,4,5,6,7,8,9,10],
    'criterion': ['gini', 'entropy']
}

CV_rfc = GridSearchCV(
    estimator=rfc,
    param_grid=param_grid,
    cv=10,
     n_jobs=-1             
)




# %%
CV_rfc.fit(X_train, Y_train)


# %%
CV_rfc.best_params_


# %%
best_rfc = RandomForestClassifier(
    criterion='gini',
    max_depth=2,
    max_features='sqrt',
    n_estimators=200,
    random_state=42
)

best_rfc.fit(X_train, Y_train)

# %%
#เช็คความแม่นยำการทำนาย
from sklearn.metrics import accuracy_score

pred = best_rfc.predict(X_test)
print("Accuracy for Random Forest on CV data:", accuracy_score(Y_test, pred))

# %%
#เช็คความ Overfit
print("Train Accuracy:", accuracy_score(Y_train, best_rfc.predict(X_train)))
print("Test Accuracy:", accuracy_score(Y_test, pred))


# %%

print("\nConfusion Matrix:\n", confusion_matrix(Y_test, pred))
print("\nClassification Report:\n", classification_report(Y_test, pred))



