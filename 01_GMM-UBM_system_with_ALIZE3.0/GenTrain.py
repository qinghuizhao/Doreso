import sys
fp=open(sys.argv[1])
m={}
for l in fp:
    l=l.strip()
    x=l.split('.')
    if x[0] in m:
        m[x[0]]=m[x[0]]+' '+l
    else:
        m[x[0]]=l

for x in m:
    print x+' '+m[x]
