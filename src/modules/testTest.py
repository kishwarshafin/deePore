
#read the reference
whole_reference = ''
with open('hg19.chr3.9mb.fa') as f:
    lines = f.readlines()
    header = 1
    for line in lines:
        if header:
            header = 0
            continue
        line = line.rstrip()
        if not line: continue
        whole_reference += line

sub_reference=''
with open('out.txt') as f:
    lines = f.readlines()
    grab = 0
    for line in lines:
        if not line: continue
        line = line.rstrip()
        if line[0]=='>':
            grab = 2
            continue
        if grab == 2:
            grab -= 1
            continue

        if grab == 1:
            line = line.replace('*','')
            sub_reference += line
            grab = 0
whole_reference = whole_reference[100000-1:110000]
sub_reference = sub_reference[:110000-100000+1]

if whole_reference == sub_reference:
    print("YEP SAME")
