import sys
fp=open(sys.argv[1])
s=''
c=0
for l in fp:
    l=l.strip()
    s=s+l+' '
    if c%3==0:
        s=s+'\n'
    c=c+1

print s
