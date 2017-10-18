import matplotlib.pyplot as plt
import numpy as np
import sys

file_name = sys.argv[1]
val_count = 0
valx = []
valy = []
testx = []
testy = []
batch_size = 217
with open(file_name, "r") as ins:
    for line in ins:
        line = line.rstrip()
        if not line:
            continue
        if line[0] == 'V':
            val_count += 1
            val_x = int(val_count) * batch_size
            val_y = float(line.split(' ')[2])
            valx.append(val_x)
            valy.append(val_y)
            print(val_x, val_y)
        else:
            split_list = line.split('\t')
            x = int(split_list[0]) * batch_size + int(split_list[1])
            y = float(split_list[2])
            testx.append(x)
            testy.append(y)

test, = plt.plot(testx, testy)
val, = plt.plot(valx, valy, 'ro-')
plt.legend([test, val], ['Train loss', 'Hold out loss'])
plt.xticks(range(0, 217 * 6, 217), ('0', '1', '2', '3', '4', '5'))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CNN Training chr1~19')
plt.savefig('CNN.png', dpi=400)