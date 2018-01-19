import shutil
import random
import numpy as np
import os
from tqdm import tqdm
import threading
import time


rep = '/media/ramin/monster/dataset/face/msceleb/MS-Celeb-1M_clean_list.txt'
from_path = '/media/ramin/monster/dataset/face/msceleb/aligned/'
to_path = '/home/ramin/Desktop/aligned-clean/'


f = open(rep, 'r')


users = dict()
sample = dict()
th_list = list()

for line in f:
	line = line.strip().split(' ')
	if line[1] not in users:
		users[line[1]] = list()
	users[line[1]].append(line[0])


user_min = 120
user_max = 500
max_thread = 10


for user in users:
	l = len(users[user])
	if l > user_max:
		print('sampled from {0}'.format(user))
		sample[user] = random.sample(users[user], user_max)
	elif l > user_min:
		print(user)
		sample[user] = users[user]
	else:
		print('skiped {0}'.format(user))
	


print('*********************************************\n\n')

print(len(sample))
l = np.array([len(sample[user]) for user in sample])
print l.mean(), l.min(), l.max()




for idx, user in enumerate(sample):
	print(idx)
	try:
		os.mkdir(os.path.join(to_path, user))
		for img in sample[user]:
			name = img.split('/')[1]
			shutil.copy2(os.path.join(from_path,img), os.path.join(to_path, user, name))
			#th = threading.Thread(target=shutil.copy2, args=(os.path.join(from_path,img), os.path.join(to_path, user, name)))
			#th.start()
			#th_list.append(th)
			#if len(th_list) > max_thread:
			#	for th in th_list:
			#		if not th.isAlive():
			#			th.join()
			#			th_list.remove(th)			
			#		time.sleep(0.3)
	except Exception,x:
		print(x)


#for th in th_list:
#	th.join()
