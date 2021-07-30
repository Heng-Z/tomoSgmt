#! usr/bin/env python3

import os

print("input tomo")
tomo = input()

dir_oritomo = '/storage/changlu/synapse202008/20200820_20200731_g2b2_65_trig/corrected_tomos/'+ tomo +'-bin8-wbp_corrected.mrc'
cmd_oritomo = 'cp '+ dir_oritomo +' ./'

dir_vesicle = '/storage/changlu/synapse202008/20200820_20200731_g2b2_65_trig/stack-out-pp/'+ tomo +'/rec/vesicle.xml'
cmd_ves = 'cp '+ dir_vesicle + ' ./'

dir_script_1 = '/home/zhenhang/sgmt_test/tomoSgmt/bin/relocal_truth.py'
dir_script_2 = '/home/zhenhang/sgmt_test/tomoSgmt/bin/proc_cmd.py'
dir_script_3 = '/home/zhenhang/sgmt_test/tomoSgmt/bin/show_all_ves.py'
dir_script_4 = '/home/zhenhang/sgmt_test/tomoSgmt/bin/json2mod.py'

cmd_script_1 = 'cp ' +dir_script_1 +' ./'
cmd_script_2 = 'cp ' +dir_script_2 +' ./demo/'
cmd_script_3 = 'cp ' +dir_script_3 +' ./demo/'
cmd_script_4 = 'cp ' +dir_script_4 +' ./demo/'

dir_area = '/storage/changlu/synapse202008/20200820_20200731_g2b2_65_trig/stack-out-pp/'+ tomo +'/rec/area.mod'
cmd_area = 'cp '+dir_area+ ' ./demo/'

os.system('mkdir demo')
os.system(cmd_oritomo)
os.system(cmd_ves)
os.system(cmd_script_1)
os.system(cmd_script_2)
os.system(cmd_script_3)
os.system(cmd_script_4)
os.system(cmd_area)
os.system('model2point ./demo/area.mod ./demo/area.point')
os.system('model2point ./demo/area.mod ./area.point')
os.system('python ./relocal_truth.py')
os.system('cp vesicle_area.xml ./demo')

