import sys
fp=open(sys.argv[1])
c=0
for l in fp:
    l=l.strip()
    if c%3!=0:
        print l
    
    c=c+1
