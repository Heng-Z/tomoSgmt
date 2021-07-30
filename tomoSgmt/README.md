### Example Code
####Predict vesicle from a missing-wedge corrected tomogram:


1. cd /storage/heng/tomoSgmt/example/test_2021Jan/t208/
2. mkdir ppXXXX
   cd ppXXXX
3. cp /storage/heng/tomoSgmt/bin/getinfo.py ./
4. python getinfo.py
5. cd demo
6. python proc_cmd.py (now model_d4_era40-100_70_1.h5 is the best one, make sure model file in /t208/)
7. copy the command and run

then, in demo dir, some useful file will be generated such as (something new listed):
	missed.mod (prediction result compare with vesicle_area.xml)
	wrong.mod 



#Some scripts new:


1. random_erase.py
used for train data(origin tomo), generate N1 2d squares and N2 3d cubes

2. relocal_truth.py
vesicle.xml file contains vesicles out of area_file, use this script to generate vesicle_area.xml file (contained in getinfo module)

3. json2mod.py
.json file in prediction result(for example, pp0365_vesicle.json) to point.mod

4. show_all_ves.py
vesicle_area.xml to vesall.mod


