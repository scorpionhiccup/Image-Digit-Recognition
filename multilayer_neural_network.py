#! /usr/bin/python

import sys
import math
import random
import matplotlib.pyplot as plt

num_hidden_units = 14
num_op_units = 10
num_ip_units = 28*28
eta = 1
threshold = 0.01
K = {}

hidden_values = [4, 8]
hidden = []
output = []
confusion_matrix = {}
num_k = 2
mean_accuracies = []

def activation(x):
	if (x<-5):
		x = -5
	ans = 1.0/(1.0 + math.e**(-1.0*x))
	return ans

def activation_derivative(x):
	temp = activation(x)
	return float(temp*(1-temp))

def test(inputs):
	z = []
	for a in xrange(len(inputs)):
		y = []
		for j in xrange(len(hidden)):
			s = 0
			for k in xrange(len(inputs[a])):
				s += float(inputs[a][k]*hidden[j][k])
			y.append(activation(s))
		z.append([])
		for i in xrange(len(output)):
			s = 0
			for j in xrange(len(hidden)):
				s += float(y[j]*output[i][j])
			z[a].append(activation(s))
	return z

def init_weights():
	global hidden, output
	for i in xrange(num_hidden_units):
		temp = []
		for j in xrange(num_ip_units):

			wt = random.uniform(-0.1/math.sqrt(num_hidden_units), 0.1/math.sqrt(num_hidden_units))
			temp.append(wt)
		hidden.append(temp)

	for i in xrange(num_op_units):
		temp = []
		for j in xrange(num_hidden_units):
			wt = random.choice([-1, 1])
			temp.append(wt)
		output.append(temp)

def train(inputs, labels):
	global hidden, output
	counter = 0
	while True and counter <= 10000:
		counter += 1

		#WEIGHTS
		for a in xrange(len(inputs)):
			#Forward Propagation
			y = []
			z = []
			netj = []
			netk = []
			old_hidden = hidden
			old_output = output
			for j in xrange(len(hidden)):
				s = 0
				for k in xrange(len(inputs[a])):
					s += 1.0*inputs[a][k]*hidden[j][k]
				netj.append(s)
				y.append(activation(s))
			
			for i in xrange(len(output)):
				s = 0
				for j in xrange(len(y)):
					s += 1.0*y[j]*output[i][j]
				netk.append(s)
				z.append(activation(s))

			#Backward Propagation
			dk = []
			for k in xrange(len(output)):
				for j in xrange(len(hidden)):
					sensitivity = 1.0*(labels[a][k]-z[k])*activation_derivative(netk[k])
					wkj = 1.0*eta*sensitivity*y[j]
					output[k][j] += wkj
				dk.append(sensitivity)

			for j in xrange(len(hidden)):
				sensitivity = 0
				for k in xrange(len(output)):
					sensitivity += 1.0*output[k][j]*dk[k]
				sensitivity *= 1.0*activation_derivative(netj[j])
				for i in xrange(len(inputs[a])):
					wji = 1.0*eta*inputs[a][i]*sensitivity 
					hidden[j][i] += wji

			flag = True
			for k in xrange(len(z)):
				if abs(labels[a][k] - z[k]) > threshold:
					flag = False

			if flag:
				break
	

def accuracy(z, labels):
	global K, num_op_units, confusion_matrix
	
	p = []
	for j in z:
		mx = 0
		for i in range(num_op_units):
			if j[mx] < j[i]:
				mx = i
		p.append(K[mx])
	z = p
	
	count = 0
	total = len(z)
	for i in xrange(total):
		if z[i] == labels[i]:
			count += 1
		confusion_matrix[str(labels[i].index(1))][str(z[i].index(1))] +=1
	
	return float(count)/total

def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(num_ip_units):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()


def plot_accuracy(accuracies):
	global num_hidden_units, num_k

	x = range(1, len(accuracies) + 1)

	plt.plot(x, accuracies, 'rx--')
	plt.xlabel('K value(s) [' + str(num_k))
	plt.ylabel('Accuracy Score')
	plt.title('No. of Hidden Units: ' + str(num_hidden_units))
	plt.draw()
	plt.show()

def plot_mean_accuracies(mean_accuracies):
	global hidden_values

	plt.plot(hidden_values, mean_accuracies, 'rx--')
	plt.xlabel('No. of Hidden Units')
	plt.ylabel('Mean Accuracy')
	plt.title('No. of Hidden Units vs. Mean Accuracy')
	plt.draw()
	plt.show()

def main():
	global K, num_op_units, hidden, output, confusion_matrix, num_k, hidden_values

	for i in range(num_op_units):
		K[i] = [0]*num_op_units
		K[i][i] = 1

	if len(sys.argv) != 2:
		print "Usage : python ann.py train-data-file test-data-file"

	convert_flag = False

	if convert_flag:
		print 'Start Conversion'

		convert("Input/train-images.idx3-ubyte", "Input/train-labels.idx1-ubyte",
	        "Input/mnist_train.csv", 60000)
		
		convert("Input/t10k-images.idx3-ubyte", "Input/t10k-labels.idx1-ubyte",
	        "Input/mnist_test.csv", 10000)

		print 'Done Converting'

	#sys.exit(1)

	f = open(sys.argv[1], 'r')
	data = [each.strip('\n') for each in f.readlines()]	

	for value in hidden_values:
		num_hidden_units = value
		accuracies = []

		print '--------------------------------------------------------------'
		print '                       Hidden Units: ' + str(value)
		print '--------------------------------------------------------------'
		
		for I in range(num_k):
			hidden = []
			output = []
			confusion_matrix = {}

			for key in range(num_op_units):
				for key2 in range(num_op_units):
					try:
						confusion_matrix[str(key)][str(key2)]=0
					except Exception, e:
						confusion_matrix[str(key)]={}
						confusion_matrix[str(key)][str(key2)] = 0
		
			#Train Neural Network
			div = len(data)/num_k
			images = [ [float(i)/255.0 for i in each.split(',')[1:]] for each in data]
			labels = [ K[i] for i in [int(each.split(',')[0]) for each in data] ]
			labels = labels[0:I*div] + labels[I*div+div:]
			images = images[0:I*div] + images[I*div+div:]
			init_weights()
			train(images, labels)
			
			#Testing Phase
			images = [ [float(i)/255.0 for i in each.split(',')[1:]] for each in data]
			labels = [ K[i] for i in [int(each.split(',')[0]) for each in data] ] 
			labels = labels[I*div:I*div+div]
			images = images[I*div:I*div+div]
			trained_output = test(images)
			
			accuracies.append(accuracy(trained_output, labels))
			print 'Accuracy Score:', accuracies[-1]
			
			if None in confusion_matrix:
				del confusion_matrix[None]

			print 'Confusion Matrix: '
			for _ in range(len(confusion_matrix)):
				print ' -----',
			print ''

			for key in sorted(confusion_matrix):
				print '|', 
				print key.ljust(4).rjust(4),
			print '|'

			for _ in range(len(confusion_matrix)):
				print ' -----',
			print ''

			for key in sorted(confusion_matrix):
				for secondary_key in sorted(confusion_matrix[key]):
					print '|', str(confusion_matrix[key][secondary_key]).ljust(4).rjust(4),
				print '|'

			for _ in range(len(confusion_matrix)):
				print ' -----',
			print ''

		plot_accuracy(accuracies)
		mean_accuracies.append(reduce(lambda x, y: x + y, accuracies) / float(len(accuracies)))

	plot_mean_accuracies(mean_accuracies)

if __name__ == "__main__":
	main()
