import os
import shutil as sh

path = 'Z:/majorProject/input/test_roi_0.15/Type_1'



import os, fnmatch
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
	            print(name)
	            os.remove(path+'/'+name)
	            result.append(os.path.join(root, name))
    return result

find('*copy.png', path)