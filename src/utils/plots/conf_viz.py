import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
array = [[8732639,    8790,     681],
 [  37777,  126803,    2214],
 [  12651,    4989,   71437]]
array = np.array(array, dtype=float)
cm = np.array(array, dtype=float)
n_categories = 3
for i in range(n_categories):
    array[i] = array[i] / array[i].sum()
print(array)
true_pos = np.diag(cm)
false_pos = np.sum(cm, axis=0) - true_pos
false_neg = np.sum(cm, axis=1) - true_pos
#print(true_pos)
#print(false_pos)
#print(false_neg)

precision = true_pos / (true_pos+false_pos)
recall = true_pos / (true_pos + false_neg)
F1 = 2 * precision * recall / (precision + recall)
print('Precision: ', precision)
print('Recall: ', recall)
print('F1: ', F1)

df_cm = pd.DataFrame(array, index=[i for i in ["Hom", "Het", "Hom-Alt"]],
                  columns=[i for i in ["Hom", "Het", "Hom-Alt"]])
plt.figure(figsize=(4*3, 3*3))
sn.heatmap(df_cm, annot=True)
plt.title('Chr20~22: Confusion Matrix')
plt.savefig('Confusion_chr20~22.png', dpi=400)


