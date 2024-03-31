import numpy as np

scores = [28, 35, 26, 32, 28, 28, 35, 34, 46, 42, 37]
mean_score = np.mean(scores)
std_dev = np.std(scores)

z_scores = [(i - mean_score) / std_dev for i in scores]
f_point = -1

f_grades = []
for i , j in zip (scores , z_scores):
    if j<f_point:
        f_grades.append(i)
    
print("avg:", mean_score)
print("standard deviation:", std_dev)
print("standard score:", z_scores)
print("F:", f_grades)