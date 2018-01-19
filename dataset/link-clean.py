
import shutil
import os 



rep = '/media/ramin/monster/dataset/face/msceleb/MS-Celeb-1M_clean_list.txt'
from_path = '/media/ramin/monster/dataset/face/msceleb/aligned/'
to_path = '/media/ramin/monster/dataset/face/msceleb/aligned-clean-2/'

files = dict()

f = open(rep, 'r')

counter = 0

for line in f:
	line = line.strip().split(' ')
	dir_, filename = line[0].split('/')
	try:
		if dir_ not in files:
			os.mkdir(os.path.join(to_path, dir_))
			files[dir_] = list()
		os.symlink(os.path.join(from_path, dir_, filename), os.path.join(to_path, dir_, filename))
		#shutil.move(os.path.join(from_path, dir_, filename), os.path.join(to_path, dir_, filename))
	except Exception,x:
		pass
	files[dir_].append(filename)
	counter +=1 
	if (counter % 1000 == 0):
		print counter
	if counter > 50000:
		break
	

