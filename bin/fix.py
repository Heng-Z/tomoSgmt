#!usr/bin/env python3
with open("missed_vesicles.point","r") as m:
	ves = m.readlines()
	#print(ves)
	vesi = []
	for i in range(len(ves)):
		#ves[i] = 'i'+ ves[i]
		idx = str(i+1)
		vesi.append(idx+' '+ves[i])

with open("missed.point", "w") as f:
	for i in vesi:
		f.write(i)
