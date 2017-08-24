import matplotlib.pyplot as plt
import numpy as np

sub_reference=''
rows = []
mylist = []
depth = 50
for i in range(depth):
    a = ''
    mylist.append(a)
with open('out.txt') as f:
    lines = f.readlines()
    line_no = 0
    row_no = 0
    length = 0
    for line in lines:
        if not line: continue
        line = line.rstrip()
        line_no += 1

        if line[0]=='>':
            line_no = 0
            for i in range(row_no, depth):
                mylist[i] += ' '*length
            row_no = 0
            continue
        if line_no == 1 or line_no == 3:
            continue

        if line_no == 2:
            length = len(line)
            sub_reference += line
        else:
            mylist[row_no]+=line
            row_no+=1
    for i in range(row_no, depth):
        mylist[i] += ' ' * length


def get_code(ch):
    if ch=='a' or ch=='A':
        return (1,0,0)
    if ch=='c' or ch=='C':
        return (0,1,0)
    if ch=='t' or ch=='T':
        return (0,0,1)
    if ch=='g' or ch=='G':
        return (1,1,0)
    if ch=='*':
        return (0,0,0)
    return (1,1,1)

print(len(sub_reference))
data = np.zeros((depth+1, len(sub_reference), 3))

for i in range(len(sub_reference)):
    data[0,i] = get_code(sub_reference[i])
for i in range(0,depth):
    for j in range(len(mylist[i])):
        code = (1,1,1)
        if mylist[i][j]=='.' or mylist[i][j]==',':
            code = get_code(sub_reference[j])
        else:
            code = get_code(mylist[i][j])

        data[i+1,j] = code
fig = plt.figure(frameon=False)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')

plt.imshow(data, interpolation='nearest')
#plt.xticks(np.arange(0.0, 2.5, 1), np.arange(0.5, 2, 0.5))
#plt.yticks(np.arange(2, -0.5, -1), np.arange(0.5, 2, 0.5))
plt.savefig('test.png')