import numpy as np
import matplotlib.pyplot as plt

ht = np.random.randint(140, 200, 100)
wt = np.random.uniform(40, 90, 100)

m_ht = ht / 100
bmi = wt / (m_ht * m_ht)

s_bmi = ['underweight', 'healthy', 'overweight', 'obese']
underweight = np.sum(bmi < 18.5)
healthy = np.sum((bmi >= 18.5) & (bmi <= 24.9))
overweight = np.sum((bmi >= 25) & (bmi <= 29.9))
obese = np.sum(bmi >= 30)

n_bmi = [underweight, healthy, overweight, obese]

# draw bar chart
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.bar(s_bmi, n_bmi)
plt.title('Bar Chart')

# draw pie chart
plt.subplot(2, 2, 2)
plt.pie(n_bmi, labels=s_bmi, autopct='%1.2f%%')
plt.title('Pie Chart')

# draw histogram
plt.subplot(2, 2, 3)
plt.hist(bmi, bins=[0,18.5, 24.9, 29.9, 40])
plt.title('Histogram')

# draw scatter plot
plt.subplot(2, 2, 4)
plt.scatter(range(len(wt)), wt, color='red', label='Weight')
plt.scatter(range(len(ht)), ht, color='blue', label='Height')
plt.xlabel('weight')
plt.ylabel('height')
plt.legend()
plt.title('Scatter Plot')

plt.tight_layout()
plt.show()
