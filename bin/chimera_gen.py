#!/usr/bin/env python3
import argparse
import json
import numpy as np

def HSV2RGB(h,s,v):
    hue=int((int(h)/60))%6
    f=h/60.0-int(h)/60
    p=v*(1-s)
    q=v*(1-f*s)
    t=v*(1-(1-f)*s)
    if hue==0:
        return str(v)+','+str(t)+','+str(p)
    if hue==1:
        return str(q)+','+str(v)+','+str(p)
    if hue==2:
        return str(p)+','+str(v)+','+str(t)
    if hue==3:
        return str(p)+','+str(q)+','+str(v)
    if hue==4:
        return str(t)+','+str(p)+','+str(v)
    if hue==5:
        return str(v)+','+str(p)+','+str(q)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--tomo_file', type=str, default=None, help='Your original tomo')
    parser.add_argument('--vesicle_json', type=str, default=None, help='Your original tomo')
    args = parser.parse_args()
    with open(args.vesicle_json) as f:
        ves = json.load(f)
    vesicle_info = ves['vesicles']

    f = open('ves_render.com','w')
    f.write('open {} \n'.format(args.tomo_file))
    mean_radius = []
    for i,vesicle in enumerate(vesicle_info):
        mean_radius.append(np.mean(np.array(vesicle['radii'])))
    mini = np.min(mean_radius)
    maxi = np.max(mean_radius)
    print('min max: ',mini,maxi)
            # np.array(vesicle['radii']),
            # np.array(vesicle['center']),
            # np.array(vesicle['evecs'])
    for i,vesicle in enumerate(vesicle_info):
        radius = np.mean(np.array(vesicle['radii']))
        cla = (radius- mini)*360/(maxi-mini)
        print(cla)
        f.write('shape sphere radius '+str(radius)+\
            ' color '+str(HSV2RGB(cla,1,1))+',1 ;\n')

        pos = np.array(vesicle['center'])
        f.write('move x '+str(pos[2])+' models #'+str(i+1)+' coord #0;\n')
        f.write('move y '+str(pos[1])+' models #'+str(i+1)+' coord #0;\n')
        f.write('move z '+str(pos[0])+' models #'+str(i+1)+' coord #0;\n')
    f.close()
