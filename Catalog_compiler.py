import numpy as np
import pandas as pd
from glob import glob
import os
import csv

def Join_searches(Path,Save):
    camps = glob(path + '*/')
    for camp in camps:
        files = glob(camp + '*.csv')
        savename = Save + 'NED_' + files[0].split('_')[2] + '.csv'
        objects = set([])
        for file in files:
            print(file)
            dataframe = pd.read_csv(file)
            data = dataframe.values
            for row in data:
                if row[2] not in objects:
                    objects.add(row[2])
                    if os.path.isfile(savename):
                            with open(savename, 'a') as csvfile:
                                spamwriter = csv.writer(csvfile, delimiter=',')
                                spamwriter.writerow(row[2:])
                    else:
                        with open(savename, 'w') as csvfile:
                            spamwriter = csv.writer(csvfile, delimiter=',')
                            spamwriter.writerow(dataframe.keys()[2:])
                            spamwriter.writerow(row[2:])
                else:
                    pass
        print('Done NED cat for C' + files[0].split('_')[2])
    return objects, 'Done NED'