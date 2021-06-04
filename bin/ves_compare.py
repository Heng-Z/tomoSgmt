#!/usr/bin/env python3
#Compare the 'correct' vesicle info with ves.json
import xml.etree.ElementTree as ET
import numpy as np
import json
import sys
# def convhull_area()
vesicle_file = sys.argv[1]
ves_xml = sys.argv[2]
binv = int(sys.argv[3])
with open(vesicle_file) as f:
    ves = json.load(f)
my_ves_list = ves['vesicles']
c = len(my_ves_list)

tree = ET.parse(ves_xml)
root = tree.getroot()
matched_pair = []
mismatched_target = []
radius_diff = []
miss = open('missed_vesicles.point','w')
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
    if ratio < 0.4 and abs(radius_diff_ratio) < 0.4 :
        match_dict = {'targe':vesid,'mine':my_ves_min['name'],'distance':ratio,'radius_diff':radius_diff_ratio}
        matched_pair.append(match_dict)
        radius_diff.append(radius_diff_ratio)
        del my_ves_list[min_dis_ind]
    else:
        mismatched_target.append({'targe':vesid,'mine':my_ves_min['name'],'distance':ratio})
        miss.write(' '.join(str(x) for x in list(xyz))+'\n')
with open('matched_pair.json','w') as f:
    f.write(json.dumps(matched_pair,indent=4, sort_keys=True))
with open('mismatched_target.json','w') as w:
    w.write(json.dumps(mismatched_target,indent=4, sort_keys=True))
miss.close()
a = len(matched_pair)
b = len(mismatched_target)

print('error rate 1:',b/(a+b))
print('error rate 2:',(c-a)/c)
print('radius diff: ',np.mean(radius_diff))



    
