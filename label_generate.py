#!/usr/bin/env python

labels = set()

inputFilename = ['train.txt', 'valid.txt', 'test.txt']

for inputfile in inputFilename:
	print 'Reading the labels from ', inputfile

	filenames = file(inputfile).readlines()

	for f in filenames:
		fname = f.strip()
		txtFilename = fname.replace('png','gt.txt')
		#print fname
		transcription = file(txtFilename).readline()
		transcription_flip = transcription.decode("utf-8")[::-1]
	
		ligs = transcription_flip.split()
		#print ligs

		for lig in ligs:
			#lig_str = str(lig.encode('utf-8'))
			lig = lig.strip()
			for character in lig:
				labels.add(character.encode('utf-8'))			
				#break
		
		#break

whitespace = u'\u0020'
labels.add(whitespace.encode('utf-8'))
#print labels

DIGITS = ''
for label in labels:
	DIGITS+=label
	#print label

#print 'Total number of labels in the dictionary: ', len(DIGITS.decode('utf-8'))








