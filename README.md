### Example Code
####Predict vesicle from a missing-wedge corrected tomogram:

cd /storage/heng/tomoSgmt/example/test_2021Jan/t208/demo

../../../../bin/mwr_vesicle.py --mwr_file ../../pp2832-bin2-bin4-5i-iter40.rec --sgmtmodel ../model_nb7-3.h5 --output_file 'pp2832_vesicle.json'  --gpuID '0,1' --render 'pp2832-vesicles.mrc' --neighbor_in 7 --neighbor_out 3 --render_in 'pp2832-vesicles-in.mrc' --output_file_in_presyn 'pp2832_vesicle_in.json'
