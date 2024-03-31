import numpy as np

ht = np.random.randint(140,200,100)
wt = np.random.uniform(40,90,100)

m_ht = ht/100
bmi = wt/(m_ht*m_ht)

print(bmi[0:10])
