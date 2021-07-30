#! usr/bin/env python3

# remove vesicles out of area set(from vesicles in vesicle.xml)

import numpy as np
import xml.etree.ElementTree as ET
import os
from tomoSgmt.bin.post_proc import *
import argparse
import json


parser = argparse.ArgumentParser(description='Process some integers')
parser.add_argument('--area_file', type=str, default='area.point', help='file which defined interested area(.point)')
parser.add_argument('--truth_file', type=str, default='vesicle.xml', help='original file for relocal vesicles(.xml)')
parser.add_argument('--new_truth_file', type=str, default='vesicle_area.xml', help='new generate file(.xml)')
args = parser.parse_args()

binv = 2
#ves_xml = 'vesicle.xml'

tree = ET.parse(args.truth_file)
root = tree.getroot()
P = get_area_points(args.area_file)
CH = Graham_scan(P)


for vesicle in root:
	xyz = np.array([vesicle[0].attrib[i] for i in ['X', 'Y', 'Z']], dtype = np.float)/binv
	center = np.delete(xyz, 2) #2d now

	if Check(CH, len(CH), center) == False:
		root.remove(vesicle)

tree.write(args.new_truth_file)
