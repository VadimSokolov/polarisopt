import sys
import fnmatch
import os
import json

def modify(json_file, key, value):
	with open(json_file, 'r+') as f:
		data = json.load(f)
		data[key] = value  # <--- add `id` value.
		f.seek(0)  # <--- should reset file position to the beginning.
		json.dump(data, f, indent=4)
		f.truncate()  # remove remaining part

def modify_old():
	f = open(sys.argv[1],'r')
	filedata = f.read()
	datalist = []

	f.close()

	for d in filedata.split(','):
		datalist.append(d.strip())

	iter = 1.0/float(sys.argv[3])

	find_str = '"' + sys.argv[2] + '" : *'
	find_match_list = fnmatch.filter(datalist,find_str)

	if len(find_match_list)==0:
		print('Could not find parameter: ', find_str)
		sys.exit()

	find_match = find_match_list[len(find_match_list)-1]

	print(find_match)
	newstr = '"' + sys.argv[2] + '" : ' + str(iter)

	newdata = filedata.replace(find_match,newstr)

	f = open(sys.argv[1],'w')
	f.write(newdata)
	f.close()