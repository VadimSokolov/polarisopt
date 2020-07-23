import sys
import fnmatch
import os


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