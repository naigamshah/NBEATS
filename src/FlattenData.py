import os
import numpy as np
import pandas as pd
import argparse

# Argument Parser
parser = argparse.ArgumentParser(description ='Search some files')   
parser.add_argument('-d', '--data', required = True, dest = 'input_file_path', action = 'store', help = 'input datafile path')
parser.add_argument('-o', '--out', required = True, dest = 'output_directory', action = 'store', help = 'output data storage directory')    
args = parser.parse_args()

#Assigning arguments to corresponding variables 
input_file_path = args.input_file_path
output_directory = args.output_directory

#1st Check
assert input_file_path[-4:]==".csv", "Your input file should be of .csv format"
output_directory = output_directory if output_directory[-1]=='/' else output_directory+'/'
print("Read input file path: " + str(os.path.abspath(input_file_path)))
print("Read output directory: " + str(os.path.abspath(output_directory)))


# Reading data from given path
print("\nReading data....")
data = pd.read_csv(input_file_path)

#2nd Check
# Checking the shape of data file
assert data.shape[1]==104, "Your input file should contain 104 columns (1-metric, 1-unit, 1-dept, 1-data, 96-entries, 1-sum, 1-avg, 1-min, 1-max), in the exact format specified"

metric_col = data.columns[0]
unit_col = data.columns[1]
dept_col = data.columns[2]


# Retrieving unique time series 
metric_unique = data.iloc[:,0].unique()
unit_unique = data.iloc[:,1].unique()
department_unique = data.iloc[:,2].unique()

print("Unique Metrics in given csv: " + str(metric_unique))
print("Unique Units in given csv: " + str(unit_unique))
print("Unique Departments in given csv: " + str(department_unique))


# File generation in required format to run NBEATS.
print("\nCreating files....")
for m in metric_unique: 
    for u in unit_unique:
        for d in department_unique:
        	u_name = os.path.basename(u)
            ids = []
            values = []
            sub_data = data.query(metric_col+' == "' + m +'" and ' + unit_col + ' == "' + u + '" and ' + dept_col + ' == ' + str(d))
            for i in range(len(sub_data)):
                ids.extend([str(sub_data.iloc[i,3]) + '_{}'.format(j) for j in range(1,97)])
                values.extend(sub_data.iloc[i,4:-4])
            flattened_data_dict = {'ID':ids, 'Value':values}     
            globals()["data_"+m +"_"+u+"_dept-"+str(d)] = pd.DataFrame(data = flattened_data_dict)
            globals()["data_"+m +"_"+u+"_dept-"+str(d)].to_csv(output_directory+"data_"+m +"_"+ u_name +"_dept-"+str(d)+".csv", index = False)
            print("Data file data_"+m +"_"+u_name+"_dept-"+str(d)+".csv created with " + str(len(sub_data)*96) + " entries.")
print(str(len(metric_unique)*len(unit_unique)*len(department_unique)) + " new files created!")

# -- End of Code --



