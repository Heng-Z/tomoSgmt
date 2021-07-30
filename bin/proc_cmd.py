#!usr/bin/env python3

#generate cmd for sgmt_predict
cmd = []
print("proc_tomo_dirc_name:(for example: pp0405)")
dirc_name = input()
print("model:(for example: model_nb7-3.h5)")
model = input()

print("python3 ../../../../../bin/mwr_vesicle.py --mwr_file ../%s-bin8-wbp_corrected.mrc --sgmtmodel ../../%s --output_file '%s_vesicle.json'  --gpuID '0,1' --render '%s-vesicles.mrc' --neighbor_in 7 --neighbor_out 3 --render_in '%s-vesicles-in.mrc' --area_file 'area.point'"%(dirc_name, model, dirc_name, dirc_name, dirc_name))
