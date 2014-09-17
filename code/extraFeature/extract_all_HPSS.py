import os 
import sys

def walk_dir(inpath,output):
    

    if not os.path.exists(output):
        os.makedirs(output)
    for item in os.listdir(inpath):
        ipath = os.path.join(inpath,item)
        opath = os.path.join(output,item)

        if os.path.isdir(ipath):
            walk_dir(ipath,opath)
        elif os.path.splitext(ipath)[1] == '.wav':
           # dirname = os.path.dirname(opath)
           # basename = os.path.basename(opath)
            
           # [name,ext] = os.path.splitext(basename)
            
           # p = os.popen('' './HPSS ' +  ipath + ' '+ dirname + \
           #      '/percussive/test/' + basename \
           #  + ' ' +dirname + '/harmonic/test/'+ basename )
            p = os.popen('' './HPSS ' + ipath + ' '+  opath )
if __name__ == '__main__':
    inp = sys.argv[1] 
    outp = sys.argv[2] 
    if os.path.exists(outp):

        
        os.system('rm -r '+ outp)
    walk_dir(inp,outp)


            
