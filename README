# Health Insurance Data Exploratory Data Analysis (EDA)

## Project Overview

This project involves performing a detailed Exploratory Data Analysis (EDA) on a health insurance dataset retrieved from Kaggle. The EDA process helps us understand the data better, identify patterns, relationships, and anomalies, and inform subsequent analysis. We use various data visualization techniques to explore the dataset, focusing on key features such as age, BMI, number of children, smoking status, gender, and region. 

## Dataset

The dataset contains information about members of a health insurance plan. It includes the following features:
- **age**: Age of the primary beneficiary.
- **sex**: Gender of the primary beneficiary.
- **bmi**: Body Mass Index, providing an understanding of body fat based on height and weight.
- **children**: Number of children/dependents covered by the insurance plan.
- **smoker**: Smoking status of the primary beneficiary.
- **region**: The beneficiary's residential area in the US (northeast, southeast, southwest, northwest).
- **charges**: The medical costs billed by the insurance.

## Installation

The project requires the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`

You can install the required libraries using pip:
```sh
pip install numpy pandas matplotlib seaborn
```

## Directory Structure

```
.
├── input
│   └── insurance.csv
├── EDA.ipynb
└── README.md
```

- **input**: Directory containing the dataset.
- **EDA.ipynb**: Jupyter notebook containing the EDA code.
- **README.md**: This file.

## Steps for EDA

### 1. Importing Libraries and Dataset

```python
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

print(os.listdir("input"))
data = pd.read_csv('input/insurance.csv')
print(data.shape)
```

### 2. Data Overview

```python
data.info()
data.describe()
data.head()
```

### 3. Pairwise Relationships

```python
plt.figure(figsize = (18, 8))
sns.pairplot(data=data, diag_kind='kde')
plt.show()
```

### 4. Age Analysis

```python
plt.figure(figsize = (18, 8))
sns.distplot(data['age'], bins=10, color='m', kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
sns.despine()
plt.show()
```

### 5. Age vs Charges

```python
plt.figure(figsize = (18, 8))
sns.scatterplot(x=data['age'], y=data['charges'], color='r')
plt.title("Age vs Charges")
sns.despine()

plt.figure(figsize = (18, 8))
sns.barplot(x='age', y='charges', data=data)
plt.title("Age vs Charges")
sns.despine()
```

### 6. Children Analysis

```python
plt.figure(figsize = (18, 8))
sns.countplot(data['children'], palette='Blues_d').set_title('Distribution of Number of Children')
plt.xlabel('Number of Children')
plt.ylabel('Count')
sns.despine()
plt.show()

plt.figure(figsize = (18, 8))
sns.boxplot(x='children', y='charges', data=data, palette='Blues_d')
plt.title('Children vs Charges')
sns.despine()
```

### 7. Gender Analysis

```python
colors = ['royalblue', 'pink']
labels = "Male", "Female"
size = (data['sex'].value_counts())
plt.pie(size, colors=colors, labels=labels, shadow=True)
plt.title('Number of Male vs Female Members')
plt.show()

plt.figure(figsize = (18, 6))
sns.violinplot(x='sex', y='charges', data=data, palette=colors, orient='v')
plt.title('Sex vs Charges')
sns.despine()
```

### 8. Smoking Status Analysis

```python
colors = ['gray', 'green']
labels = "Smoker", "Non-Smoker"
size = (data['smoker'].value_counts())
plt.pie(size, colors=colors, labels=labels, shadow=True)
plt.title('Number of Smoking vs Non-Smoking Members')
plt.show()

plt.figure(figsize = (18, 6))
sns.scatterplot(x='age', y='charges', hue='smoker', palette=colors, data=data)
plt.title('Charges vs Age Filtered by Smoking Status')
sns.despine()
```

### 9. BMI Analysis

```python
plt.figure(figsize = (18, 8))
sns.scatterplot(x='age', y='charges', hue='bmi', data=data)
plt.title('Charges vs Age filtered by BMI')
sns.despine()

data['BMI_smoker'] = 'default'
data.loc[(data.bmi >= 30) & (data.smoker == 'yes'), 'BMI_smoker'] = 'High_BMI_smoker'
data.loc[(data.bmi >= 30) & (data.smoker == 'no'), 'BMI_smoker'] = 'High_BMI_no_smoker'
data.loc[(data.bmi < 30) & (data.smoker == 'yes'), 'BMI_smoker'] = 'Normal_BMI_smoker'
data.loc[(data.bmi < 30) & (data.smoker == 'no'), 'BMI_smoker'] = 'Normal_BMI_no_smoker'

plt.figure(figsize = (18, 10))
sns.scatterplot(x='age', y='charges', hue='BMI_smoker', palette='pastel', data=data, style='BMI_smoker')
plt.title('Charges vs Age Filtered by BMI_smoker')
sns.despine()
```

### 10. Region Analysis

```python
plt.figure(figsize = (18, 10))
sns.countplot(data['region'], palette='YlOrBr').set_title('Distribution of Members Living in Different Regions')
plt.xlabel('Regions')
plt.ylabel('count')
sns.despine()
plt.show()

plt.figure(figsize = (18, 8))
sns.boxplot(x='region', y='charges', data=data, palette='YlOrBr')
plt.title('Region vs Charges')
sns.despine()
```

### 11. Correlation Heatmap

```python
data['male'] = data['sex'].replace({'female': 0, 'male': 1})
data['smoker'] = data['smoker'].replace({'no': 0, 'yes': 1})

data2 = data.copy()
data2.drop(['sex'], axis=1, inplace=True)

f, ax = plt.subplots(figsize=(10, 10))
corr = data2.corr()
sns.heatmap(corr, linewidths=.5, annot=True, square=True, ax=ax)
plt.show()
```

## Conclusion

The EDA reveals several key insights:
- Smoking status is the most correlated with charges.
- Age, BMI, number of children, and sex also influence charges.
- High BMI and smoking together significantly increase insurance claims.
- Regional differences in charges exist but are not dramatic.

## References

1. [Seaborn Pairplot](https://seaborn.pydata.org/generated/seaborn.pairplot.html)
2. [Kernel Density Estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation)
3. [Violin Plot](https://en.wikipedia.org/wiki/Violin_plot)

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

Feel free to clone the repository, explore the data, and contribute to the project!
