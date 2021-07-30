if __name__ == "__main__":
	
	import argparse
	import os


	print("input tomo")
	tomo = input()
	parser = argparse.ArgumentParser(description='Process some integers.')
	json_file = tomo +'-bin8-wbp_corrected-vesicle-in-area.json'
	parser.add_argument('--json_file', type=str, default=json_file, help='.json file')
	args = parser.parse_args()

	with open(args.json_file, "r") as t:
		data = eval(t.read()).get('vesicles')
	center = []
	for i in range(len(data)):
		center.append(data[i].get('center'))
	for i in range(len(center)):
		center[i][0], center[i][1], center[i][2] = center[i][2], center[i][1], center[i][0]	
	
	with open("point.txt", "w") as p:
		contour = 0
		for i in center:
			contour = contour+1
			p.write('{} '.format(contour))
			for j in i:
				p.write(str(j))
				p.write(' ')
			p.write('\n')

	cmd = 'point2model point.txt point.mod'
	with open("point.txt", "r") as p:
		os.system(cmd)
