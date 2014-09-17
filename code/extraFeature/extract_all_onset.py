import os 

def walk_dir(inpath,output):
    

    if not os.path.exists(output):
        os.makedirs(output)
    for item in os.listdir(inpath):
        ipath = os.path.join(inpath,item)
        opath = os.path.join(output,item)

        if os.path.isdir(ipath):
            walk_dir(ipath,opath)
        elif os.path.splitext(ipath)[1] == '.wav':
            p = os.popen('' './standard_onsetrate ' +  ipath  + ' > ' + opath +'.txt')

if __name__ == '__main__':
    inp = '/media/tiger/data/splited/PandoraSubset/train'
    outp = '/media/tiger/data/out/PandoraSubset/onset/train/'
    if os.path.exists(outp):
        os.system('rm -r '+ outp)
    walk_dir(inp,outp)


            
