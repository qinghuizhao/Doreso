import os
import os.path
import shutil
import random

def SelectTest_Val(rootdir,num):
	allfile = os.listdir(rootdir)
	random.shuffle(allfile)
	testfile = []
	valfile = []
	for i in range(num):
		testfile.append(allfile(i))


	for i in range(num):
		valfile.append(allfile(i+num))

	for i in allfile:
		if i in testfile or i in valfile:
			allfile.remove(i)

	parent, dirnames, filenames = os.walk(rootdir)

	

	for i in range(num):
		p,f = os.path.split(testfile[i])
		test_outputpath = os.path.join(parent,"test/",f)
		if (os.path.exists(test_outputpath) == False):
			dir1 = os.path.dirname(test_outputpath)
    		os.makedirs(dir1) 
		shutil.copyfile(testfile[i],test_outputpath)
		

	for i in range(num):
		p,f = os.path.split(valfile[i])
		val_outputpath = os.path.join(parent,"val/",f)
		if (os.path.exists(val_outputpath) == False):
			dir1 = os.path.dirname(val_outputpath)
    		os.makedirs(dir1) 
		shutil.copyfile(valfile[i],val_outputpath)

	for i in range(len(allfile)):
		p,f = os.path.split(allfile[i])
		train_outputpath = os.path.join(parent,"train/",f)
		if (os.path.exists(train_outputpath) == False):
			dir1 = os.path.dirname(train_outputpath)
    		os.makedirs(dir1) 
		shutil.copyfile(valfile[i],train_outputpath)




#if __name__ == '__main__':
#	pass
	#SelectTest_Val(root,10)
		