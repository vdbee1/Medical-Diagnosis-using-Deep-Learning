import cv2
import os

type1 = "Z:/minorProject/cervical data/train/type 1"
type2 = "Z:/minorProject/cervical data/train/type 2"
type3 = "Z:/minorProject/cervical data/train/type 3"
type1Test = "Z:/minorProject/cervical data/test/type 1"
type2Test = "Z:/minorProject/cervical data/test/type 2"
type3Test = "Z:/minorProject/cervical data/test/type 3"
type1GreenChannelTrain = 'Z:/minorProject/cervical data/greenChannelTrain/type 1'
type2GreenChannelTrain = 'Z:/minorProject/cervical data/greenChannelTrain/type 2'
type3GreenChannelTrain = 'Z:/minorProject/cervical data/greenChannelTrain/type 3'
type1GreenChannelTest = 'Z:/minorProject/cervical data/greenChannelTest/type 1'
type2GreenChannelTest = 'Z:/minorProject/cervical data/greenChannelTest/type 2'
type3GreenChannelTest = 'Z:/minorProject/cervical data/greenChannelTest/type 3'
# make additional path for additional folders


for file in os.listdir(type3Test):
	#print(type1+'/'+file)
	filePath = type3Test+'/'+file
	img = cv2.imread(filePath)
	i = img
	i[:,:,0]=0
	i[:,:,2]=0
	cv2.imwrite(type3GreenChannelTest+"/"+file,i)


