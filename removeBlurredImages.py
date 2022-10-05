import pandas as pd
import os

excelPath = "blurred.xlsx"
pathTrainTyp1 = 'Z:/majorProject/input/train_roi_0.15/Type_1/'
pathTrainTyp2 = 'Z:/majorProject/input/train_roi_0.15/Type_2/'
pathTrainTyp3 = 'Z:/majorProject/input/train_roi_0.15/Type_3/'

pathAddTyp1 = "Z:/majorProject/input/additional_roi_0.15/Type_1/"
pathAddTyp2 = "Z:/majorProject/input/additional_roi_0.15/Type_2/"
pathAddTyp3 = "Z:/majorProject/input/additional_roi_0.15/Type_3/"

pathTest = "Z:/majorProject/input/test_roi_0.15/"

ext = '.png'

df = pd.read_excel(excelPath)
df.astype({'Name': 'int32'})

for index, Name in enumerate(df['Name']):
	Name = str(Name)
	if df.iloc[index]['Additional Typ1'] == "Yes":
		if os.path.exists(pathAddTyp1 + Name + ext):
			os.remove(pathAddTyp1 + Name + ext)
			print(f"Removed file {Name} from folder Additional Typ1 !!!")
		else:
			print(f"Nothing to remove addtyp1; count = {index}; name = {Name} ")
	elif df.iloc[index]['Additional Typ2'] == "Yes":
		if os.path.exists(pathAddTyp2 + Name + ext):
			os.remove(pathAddTyp2 + Name + ext)
			print(f"Removed file {Name} from folder Additional Typ2 !!!")
		else:
			print(f"Nothing to remove addtyp2; count = {index}; name = {Name}; path = {pathAddTyp2 + Name + ext} ")
	elif df.iloc[index]['Additional Typ3'] == "Yes":
		if os.path.exists(pathAddTyp3 + Name + ext):
			os.remove(pathAddTyp3 + Name + ext)
			print(f"Removed file {Name} from folder Additional Typ3 !!!")
		else:
			print(f"Nothing to remove addtyp3; count = {index}; name = {Name} ")
	elif df.iloc[index]['TrainTyp1'] == "Yes":
		if os.path.exists(pathTrainTyp1 + Name + ext):
			os.remove(pathTrainTyp1 + Name + ext)
			print(f"Removed file {Name} from folder TrainTyp1 !!!")
		else:
			print(f"Nothing to remove traintyp1; count = {index}; name = {Name} ")
	elif df.iloc[index]['TrainTyp2'] == "Yes":
		if os.path.exists(pathTrainTyp2 + Name + ext):
			os.remove(pathTrainTyp2 + Name + ext)
			print(f"Removed file {Name} from folder TrainTyp2 !!!")
		else:
			print(f"Nothing to remove traintyp2; count = {index}; name = {Name} ")
	elif df.iloc[index]['TrainTyp3'] == "Yes":
		if os.path.exists(pathTrainTyp3 + Name + ext):
			os.remove(pathTrainTyp3 + Name + ext)
			print(f"Removed file {Name} from folder from folder TrainTyp3!!!")
		else:
			print(f"Nothing to remove traintyp3; count = {index}; name = {Name} ")
	elif df.iloc[index]['Test'] == "Yes":
		if os.path.exists(pathTest + Name + ext):
			os.remove(pathTest + Name + ext)
			print(f"Removed file {Name} from folder Test !!!")
		else:
			print(f"Nothing to remove test; count = {index}; name = {Name} and path = {pathTest+Name+ext} ")
