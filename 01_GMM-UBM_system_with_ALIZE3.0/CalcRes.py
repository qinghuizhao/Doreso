import sys
fp=open(sys.argv[1])
m={}
for l in fp:
    l=l.strip()
    s=l.split(' ')
    if s[3] in m:
        m[s[3]]=m[s[3]]+','+l
    else:
        m[s[3]]=l

nCount=0
nRight=0
for x in m:
    s=m[x].split(',')
    nCount=nCount+1
    fMax=-10000
    RecRes=''
    for y in s:
        z=y.split(' ')
        if fMax<(float)(z[4]):
            fMax=(float)(z[4])
            RecRes=z[1]

    if RecRes in x:
        nRight=nRight+1


print nRight
