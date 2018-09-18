from Zoo_upload_tools import *

permit = ['Unknown', 'G', 'QSO', 'R', 'IrS', 'WDStar', 'SN', 'XrayS']

prohibit = ['Star']

set_name = 'c17_test'

path = '/export/maipenrai2/skymap/brad/KEGS/Data/K2BS/Results/c17/'
save = '/export/maipenrai2/skymap/brad/KEGS/Data/K2BS/Results/c17/Zoo'

Zoo_upload(set_name, path, permit, prohibit, save)