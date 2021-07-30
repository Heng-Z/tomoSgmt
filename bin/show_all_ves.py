#! usr/bin/env python3

# show all vesicles in vesicle.xml file

import xml.etree.ElementTree as ET
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--vesicle_file', type=str, default='vesicle_area.xml', help='vesicle.xml/vesicle_area.xml')
args = parser.parse_args()

binv = 2
ves_xml = args.vesicle_file
s = 'point2model vesall.point vesall.mod'
tree = ET.parse(ves_xml)
root = tree.getroot()

with open("vesall.point", "w") as v:
	for vesicle in root:
		vesid = vesicle.attrib['vesicleId'] # str
		xyz = np.array([vesicle[0].attrib[i] for i in ['X','Y','Z']],dtype=np.float)/binv
		zyx = np.flip(xyz)
		v.write(' '.join(str(x) for x in list(xyz))+'\n')

os.system(s)

