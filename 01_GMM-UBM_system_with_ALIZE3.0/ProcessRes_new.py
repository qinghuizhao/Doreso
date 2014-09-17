import sys
fp=open(sys.argv[1])
m={}
x=''
c=0
for l in fp:
    l=l.strip()
    s=l.split(' ')
    if s[3] in m:
        m[s[3]]=m[s[3]]+','+l
        x=x+' '+s[4]
    else:
        m[s[3]]=l
        if c>1:
            print x
        x=s[4]
    c=c+1

print x
