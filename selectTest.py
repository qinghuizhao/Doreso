import random
import os
import shutil

def selectTest(inputPath,outputPath,num):
	for  parent, dirnames, filenames  in  os.walk(inputPath):
     # case 1: 
     	# for  dirname  in  dirnames:
      #    	print  ( " parent is: "   +  parent)
      #    	print  ( " dirname is: "   +  dirname)
     # case 2 
     		for  filename  in  filenames:
	         	print  ( " parent is: "   +  parent)
	         	print  ( " filename with full path : "   +  os.path.join(parent, filename))
	         	filenamewithfull = os.path.join(parent, filename)
	         	seed = random.randrange(10)
	         	if seed == 0 :
	         		outputFolder = outputPath + "test/"
	         		if (os.path.exists(outputFolder) == False):
	    				os.makedirs(outputFolder) 
	         		output = os.path.join(outputFolder,filename)
	         		shutil.copyfile(filenamewithfull,output)
	     		elif seed == 1:
	     			outputFolder = outputPath + "val/"
	     			if (os.path.exists(outputFolder) == False):
	    				os.makedirs(outputFolder) 
	         		output = os.path.join(outputFolder,filename)
	         		shutil.copyfile(filenamewithfull,output)
	         	else :
	         		outputFolder = outputPath + "train/"
	         		if (os.path.exists(outputFolder) == False):
	    				os.makedirs(outputFolder) 
	         		output = os.path.join(outputFolder,filename)
	         		shutil.copyfile(filenamewithfull,output)

if __name__ == '__main__':
	selectTest('/Users/xinquanzhou/Downloads/dukto','/Users/xinquanzhou/Downloads/dukto/new',3)