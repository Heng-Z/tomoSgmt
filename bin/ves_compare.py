#!/usr/bin/env python3
#Compare the 'correct' vesicle info with ves.json
import xml.etree.ElementTree as ET
import numpy as np
import json
import sys
import settings
import os
from math import floor

# def convhull_area()
vesicle_file = settings.vesicle_file
ves_xml = settings.ves_xml
binv = int(settings.binv)
with open(vesicle_file) as f:
    ves = json.load(f)
my_ves_list = ves['vesicles']
c = len(my_ves_list)

tree = ET.parse(ves_xml)
root = tree.getroot()
matched_pair = []
mismatched_target = []
radius_diff = []

miss_file = '../example/test_2021Jan/t208/'+settings.testtomo+'/demo/missed_vesicle.point'
model_file = '../example/test_2021Jan/t208/'+settings.testtomo+'/demo/missed.mod'
match_file = '../example/test_2021Jan/t208/'+settings.testtomo+'/demo/matched_pair.json'
mismatch_file = '../example/test_2021Jan/t208/'+settings.testtomo+'/demo/mismatched_target.json'
point_file = '../example/test_2021Jan/t208/'+settings.testtomo+'/demo/wrong.point'
mod_file = '../example/test_2021Jan/t208/'+settings.testtomo+'/demo/wrong.mod'

miss = open(miss_file,'w')
print(root.tag,root.attrib)
for vesicle in root:
    vesid = vesicle.attrib['vesicleId'] # str
    # print(vesid)
    xyz = np.array([vesicle[0].attrib[i] for i in ['X','Y','Z']],dtype=np.float)/binv
    zyx = np.flip(xyz)
    try:
        radius = np.array(vesicle[1].attrib['r'],dtype=np.float)/binv
    except:
        break
    distance_to_allleft = []
    for my_vesid,my_ves in enumerate(my_ves_list):
        distance_to_allleft.append(np.linalg.norm(zyx-np.array(my_ves['center'])))

    min_dis_ind = np.argmin(distance_to_allleft)
    my_ves_min = my_ves_list[min_dis_ind]
    ratio = distance_to_allleft[min_dis_ind]/np.mean(my_ves_min['radii'])
    # print(ratio)
    radius_diff_ratio = np.abs(np.mean(my_ves_min['radii'])-radius)/radius
    if ratio < 0.6 and abs(radius_diff_ratio) < 0.4 : #origin radio check: radio<0.4, some right-predicted vesicles will be rejected
        #match_dict = {'targe':vesid,'mine':my_ves_min['name'],'distance':ratio,'radius_diff':radius_diff_ratio}
        match_dict = {'targe':vesid,'mine':my_ves_min['name'],'distance':ratio,'radius_diff':radius_diff_ratio, 'radius_predict':my_ves_min['radii'], 'center':xyz.tolist()}
        matched_pair.append(match_dict)
        radius_diff.append(radius_diff_ratio)
        del my_ves_list[min_dis_ind]
    else:
        mismatched_target.append({'targe':vesid,'mine':my_ves_min['name'],'distance':ratio})
        miss.write(' '.join(str(x) for x in list(xyz))+'\n')

with open(match_file, 'w') as f:
    f.write(json.dumps(matched_pair,indent=6, sort_keys=True))
with open(mismatch_file, 'w') as w:
    w.write(json.dumps(mismatched_target,indent=4, sort_keys=True))

wrong = []
with open(point_file, 'w') as e:
    for my_vesid, my_ves in enumerate(my_ves_list):
        wrong.append(my_ves['center'])
    for i in range(len(wrong)):
        wrong[i][0],wrong[i][1],wrong[i][2] = wrong[i][2],wrong[i][1],wrong[i][0]
        for j in wrong[i]:
            e.write(str(floor(j))+' ')
        e.write('\n')

cmd_wrongfile = 'point2model '+ point_file +' '+ mod_file
os.system(cmd_wrongfile)

miss.close()
a = len(matched_pair)
b = len(mismatched_target)

print('error rate 1:',b/(a+b))
print('error rate 2:',(c-a)/c)
print('radius diff: ',np.mean(radius_diff))

if b != 0:
    cmd_mod = 'point2model '+ miss_file + ' ' +model_file
    os.system(cmd_mod)
else:
    print("No vesicle not find")


    
