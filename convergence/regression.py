#!/usr/bin/python
# Filename: regression.py

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt  # noqa: E402
# import sys
import os.path  # noqa: E402
import math  # noqa: E402
import numpy as np  # noqa: E402
import smtplib  # noqa: E402
from email.mime.text import MIMEText  # noqa: E402
from email.mime.image import MIMEImage  # noqa: E402
from email.mime.multipart import MIMEMultipart  # noqa: E402


# if (len(sys.argv) < 3):
# 	print "Usage %s standard_file simulated_file"%sys.argv[0]
# 	sys.exit(-1)


# standard_file = sys.argv[1]
# simulated_file = sys.argv[2]

def regression(standard_dir, simulated_dir):
	standard_file = standard_dir + '/summary.csv'
	simulated_file = simulated_dir + '/summary.csv'
	if not os.path.isfile(standard_file):
		print("ERROR: \'%s\' does not exist" % standard_file)
		return
	if not os.path.isfile(simulated_file):
		print("ERROR: \'%s\' does not exist" % simulated_file)
		return

	email_list = './email_list.txt'

	std = np.zeros(14400)
	sim = np.zeros(14400)

	inn_max = 10000
	with open(standard_file) as fh:
		fh.readline()  # slip the header
		for line in fh:
			s = line.split(',')
			time = list(map(int, s[0].split(':')))
			inn = int(s[4])
			sec = 3600*time[0] + 60*time[1] + time[2]
			std[int(sec/6)] = inn
			inn_max = max(inn_max, inn)
	with open(simulated_file) as fh:
		fh.readline()  # slip the header
		for line in fh:
			s = line.split(',')
			time = list(map(int, s[0].split(':')))
			inn = int(s[4])
			sec = 3600*time[0] + 60*time[1] + time[2]
			sim[int(sec/6)] = inn
			inn_max = max(inn_max, inn)

	inn_max = ((inn_max / int(10000)) * 10000) + 10000
			
	a = sum(std - sim)/sum(sim)
	
	mape = a*100.0

	t = np.arange(0., 14400, 1)
	fig, ax = plt.subplots()
	# plt.plot(t, std, 'g--', t, sim, 'r-')
	plt.title('Std vs Sim - MAPE=' + str(mape))
	ax.plot(t, std, 'g--', label=standard_file)
	ax.plot(t, sim, 'r-', label=simulated_file)
	ylim = ax.get_xlim()
	ax.set_ylim(ylim[0], inn_max)
	ax.legend(loc='upper left', shadow=True)
	plot_file = simulated_dir + '/in_network.png'
	plt.savefig(plot_file)

	mape_threshold = 2.0
	if math.fabs(mape) < mape_threshold:
		print('MAPE %s is below the allowed threshold of %s.' % (str(mape), str(mape_threshold)))
		return
	
	print('WARNING: MAPE of %s is greater than the allowed threshold of %s!' % (str(mape), str(mape_threshold)))
		
	# you = []
	if os.path.isfile(email_list) is not True:
		print('The email recipient list file \'%s\' does not exist' % email_list)
		return
	
	with open(email_list) as fh:
		you = fh.read().splitlines()
		
	if len(you) < 1:
		print("There are no email recipients specified")
		return
		
	print("Email will be sent to:")
	print(you)
		
	me = "polaris_testing@anl.gov"
	# you = 'rweimer@anl.gov'
	msg = MIMEMultipart()
	msg['Subject'] = 'Polaris Testing Results'
	msg['From'] = me
	msg['To'] = ", ".join(you)
	msg.preamble = 'MAPE is %s' % str(a*100.0)
	fp = open(plot_file, 'rb')
	img = MIMEImage(fp.read())
	fp.close()
	msg.attach(img)

	text_part = MIMEText('Hi there!\nMAPE  for in network data for given data compared to simulated data is  %s percent.\nSee plot attached.\n' % str(mape), 'plain')
	msg.attach(text_part)

	s = smtplib.SMTP('mailhost.anl.gov', 25)
	s.sendmail(me, you, msg.as_string())
	s.quit()
