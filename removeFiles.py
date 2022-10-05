import pandas as pd
import os

csvPath = "D:/Intel_data/removed_files.csv"
pathAdd1 = "Z:/majorProject/input/additional/Type_1/"
pathAdd2 = "Z:/majorProject/input/additional/Type_2/"
pathAdd3 = "Z:/majorProject/input/additional/Type_3/"

pathAddR1 = "Z:/majorProject/input/additional_resized/Type_1/"
pathAddR2 = "Z:/majorProject/input/additional_resized/Type_2/"
pathAddR3 = "Z:/majorProject/input/additional_resized/Type_3/"

pathAddROI1 = "Z:/majorProject/input/additional_roi_0.15/Type_1/"
pathAddROI2 = "Z:/majorProject/input/additional_roi_0.15/Type_2/"
pathAddROI3 = "Z:/majorProject/input/additional_roi_0.15/Type_3/"


df = pd.read_csv(csvPath)
for index, fileName in enumerate(df['filename']):
	if df.iloc[index]['old_label'] == "Type_1":
		if os.path.exists(pathAdd1 + fileName):
			os.remove(pathAdd1 + fileName)
			os.remove(pathAddR1 + fileName)
			os.remove(pathAddROI1 + fileName)
			print(f"Removed file {fileName} !!!")
		else:
			print(f"Nothing to remove; count = {index} ")
	elif df.iloc[index]['old_label'] == "Type_2":
		if os.path.exists(pathAdd2 + fileName):
			os.remove(pathAdd2 + fileName)
			os.remove(pathAddR2 + fileName)
			os.remove(pathAddROI2 + fileName)
			print(f"Removed file {fileName} !!!")
		else:
			print(f"Nothing to remove; count = {index} ")
	elif df.iloc[index]['old_label'] == "Type_3":
		if os.path.exists(pathAdd3 + fileName):
			os.remove(pathAdd3 + fileName)
			os.remove(pathAddR3 + fileName)
			os.remove(pathAddROI3 + fileName)
			print(f"Removed file {fileName} !!!")
		else:
			print(f"Nothing to remove; count = {index} ")

df = pd.read_csv(csvPath)
for index, fileName in enumerate(df['filename']):
	if df.iloc[index]['Additional Typ1'] == "Yes":
		if os.path.exists(pathAdd1 + fileName):
			os.remove(pathAdd1 + fileName)
			#os.remove(pathAddR1 + fileName)
			#os.remove(pathAddROI1 + fileName)
			print(f"Removed file {fileName} !!!")
		else:
			print(f"Nothing to remove; count = {index} ")
	elif df.iloc[index]['Additional Typ2'] == "Yes":
		if os.path.exists(pathAdd2 + fileName):
			os.remove(pathAdd2 + fileName)
			#os.remove(pathAddR2 + fileName)
			#os.remove(pathAddROI2 + fileName)
			print(f"Removed file {fileName} !!!")
		else:
			print(f"Nothing to remove; count = {index} ")
	elif df.iloc[index]['Additional Typ3'] == "Yes":
		if os.path.exists(pathAdd3 + fileName):
			os.remove(pathAdd3 + fileName)
			#os.remove(pathAddR3 + fileName)
			#os.remove(pathAddROI3 + fileName)
			print(f"Removed file {fileName} !!!")
		else:
			print(f"Nothing to remove; count = {index} ")
	elif df.iloc[index]['TrainTyp1'] == "Yes":
		if os.path.exists(pathAdd3 + fileName):
			os.remove(pathAdd3 + fileName)
			#os.remove(pathAddR3 + fileName)
			#os.remove(pathAddROI3 + fileName)
			print(f"Removed file {fileName} !!!")
		else:
			print(f"Nothing to remove; count = {index} ")
	elif df.iloc[index]['TrainTyp2'] == "Yes":
		if os.path.exists(pathAdd3 + fileName):
			os.remove(pathAdd3 + fileName)
			#os.remove(pathAddR3 + fileName)
			#os.remove(pathAddROI3 + fileName)
			print(f"Removed file {fileName} !!!")
		else:
			print(f"Nothing to remove; count = {index} ")