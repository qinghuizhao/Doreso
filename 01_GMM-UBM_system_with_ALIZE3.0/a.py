import sys
fp=open(sys.argv[1])
for l in fp:
    l=l.strip()
    print l + ' '+l
