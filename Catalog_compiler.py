import numpy as np
import pandas as pd
from glob import glob
import os
import csv

def Join_searches(Path,Save):
    files = glob(Path + '*.csv')
    savename = Save + 'NED_' + files[0].split('_')[2] + '.csv'
    objects = []
    for file in files:
        dataframe = pd.read_csv(file)
        data = dataframe.values
        for row in data:
            if row[1] not in objects:
                objects += [row[1]]
                if os.path.isfile(savename):
                        with open(savename, 'a') as csvfile:
                            spamwriter = csv.writer(csvfile, delimiter=',')
                            spamwriter.writerow(row)
                else:
                    with open(savename, 'w') as csvfile:
                        spamwriter = csv.writer(csvfile, delimiter=',')
                        spamwriter.writerow(dataframe.keys())
                        spamwriter.writerow(row)
            else:
                pass
    return 'Done NED cat for C' + files[0].split('_')[2]