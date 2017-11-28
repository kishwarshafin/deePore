import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns
sns.set(color_codes=True)

file_name = sys.argv[1]
val_count = 0
valx = []
valy = []
testx = []
testy = []
batch_size = 469
epoch = 2
with open(file_name, "r") as ins:
    for line in ins:
        line = line.rstrip()
        if not line:
            continue
        if line[0] == 'T':
            val_count += 1
            val_x = int(val_count) * batch_size
            val_y = float(line.split(' ')[2])
            valx.append(val_x)
            valy.append(val_y)
            #print(val_x, val_y)
        else:
            split_list = line.split('\t')
            x = int(split_list[0]) * batch_size + int(split_list[1])
            y = float(split_list[2])
            testx.append(x)
            testy.append(y)

test, = plt.plot(testx, testy)
val, = plt.plot(valx, valy, 'ro-')
plt.legend([test, val], ['Train loss', 'Hold out loss'])

gap_in_ticks = 1
x_ticks = ()
for i in range(0, epoch+1, gap_in_ticks):
    x_ticks = x_ticks + (str(i),)
plt.ylim([0, 0.000002])
print(x_ticks)
plt.xticks(range(0, batch_size * (epoch+1), batch_size*gap_in_ticks), x_ticks)
plt.xlabel('Epoch')
plt.ylabel('Loss')

# plt.title('CNN Training chr1~19')
plt.savefig(file_name+'-viz'+'.png', dpi=400)