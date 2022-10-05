import pandas as pd
import os
import shutil as sh

csvPath = "D:/Intel_data/fixed_labels_v2.csv"

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
	if df.iloc[index]['old_label'] == 'Type_1':
		if os.path.exists(pathAdd1+fileName):
			typ = df.iloc[index]['new_label']
			if typ == "Type_2":
				sh.move(src=pathAdd1 + fileName, dst=pathAdd2 + fileName)
				sh.move(src=pathAddR1 + fileName, dst=pathAddR2 + fileName)
				sh.move(src=pathAddROI1 + fileName, dst=pathAddROI2 + fileName)
			elif typ == "Type_3":
				sh.move(src=pathAdd1 + fileName, dst=pathAdd3 + fileName)
				sh.move(src=pathAddR1 + fileName, dst=pathAddR3 + fileName)
				sh.move(src=pathAddROI1 + fileName, dst=pathAddROI3 + fileName)
			else :
				print("Error marker 1")
			print(f"File {fileName} in correct folder !!!")
	elif df.iloc[index]['old_label'] == 'Type_2':
		if os.path.exists(pathAdd2+fileName):
			typ = df.iloc[index]['new_label']
			if typ == "Type_1":
				sh.move(src=pathAdd2 + fileName, dst=pathAdd1 + fileName)
				sh.move(src=pathAddR2 + fileName, dst=pathAddR1 + fileName)
				sh.move(src=pathAddROI2 + fileName, dst=pathAddROI1 + fileName)
			elif typ == "Type_3":
				sh.move(src=pathAdd2 + fileName, dst=pathAdd3 + fileName)
				sh.move(src=pathAddR2 + fileName, dst=pathAddR3 + fileName)
				sh.move(src=pathAddROI2 + fileName, dst=pathAddROI3 + fileName)
			else :
				print("Error marker 2")
		else:
			print(f"File {fileName} in correct folder !!!")
	elif df.iloc[index]['old_label'] == 'Type_3':
		if os.path.exists(pathAdd3+fileName):
			typ = df.iloc[index]['new_label']
			if typ == "Type_2":
				sh.move(src=pathAdd3+fileName, dst=pathAdd2+fileName)
				sh.move(src=pathAddR3+fileName, dst=pathAddR2+fileName)
				sh.move(src=pathAddROI3+fileName, dst=pathAddROI2+fileName)
			elif typ == "Type_1":
				sh.move(src=pathAdd3 + fileName, dst=pathAdd1 + fileName)
				sh.move(src=pathAddR3 + fileName, dst=pathAddR1 + fileName)
				sh.move(src=pathAddROI3 + fileName, dst=pathAddROI1 + fileName)
			else :
				print("Error marker 3")
		else:
			print(f"File {fileName} in correct folder !!!")
	else:
		print(f"No case satisfied for file {fileName} !!!")



