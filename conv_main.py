import numpy as np
import gzip
import pickle
from time import time
from numba import cuda, float32

epochs = 10
neurons1 = 32
neurons2 = 16
learning_rate = 1e-4

np.random.seed(1)

def saveWeights(wts):
	for num in range(len(wts)):
		with open('conv_weights/' + str(num+1) + '.pkl', 'wb') as f:
			pickle.dump(wts[num], f, pickle.HIGHEST_PROTOCOL)

def loadWeight(name):
	with open(name, 'rb') as f:
		return pickle.load(f)			

def convertToOneHot(vector, num_classes=None):
	assert isinstance(vector, np.ndarray)
	assert len(vector) > 0

	if num_classes is None:
		num_classes = np.max(vector)+1
	else:
		assert num_classes > 0
		assert num_classes >= np.max(vector)

	result = np.zeros(shape=(len(vector), num_classes))
	result[np.arange(len(vector)), vector] = 1
	return result.astype(int)

def getMnistData():
	with gzip.open('train-images-idx3-ubyte.gz', 'rb') as f:
		data = np.frombuffer(f.read(), np.uint8, offset=16)
	data = data.reshape(-1, 28, 28, 1)
	data = np.divide(data, 256)
	
	with gzip.open('train-labels-idx1-ubyte.gz', 'rb') as f:
		labels = np.frombuffer(f.read(), np.uint8, offset=8)
	labels = convertToOneHot(labels, 10)
	labels = labels.reshape(60000, 10, 1)
	
	with gzip.open('t10k-images-idx3-ubyte.gz', 'rb') as f:
		testdata = np.frombuffer(f.read(), np.uint8, offset=16)
	testdata = testdata.reshape(-1, 28, 28, 1)
	testdata = np.divide(testdata, 256)
	
	with gzip.open('t10k-labels-idx1-ubyte.gz', 'rb') as f:
		testlabels = np.frombuffer(f.read(), np.uint8, offset=8)
	testlabels = convertToOneHot(testlabels, 10)
	testlabels = testlabels.reshape(10000, 10, 1)	
	
	return data, labels, testdata, testlabels

def reLU(x):
	return np.maximum(0,x)	

def reLU_back(pro,o_grad):
	o_grad[pro < 0] = 0
	return o_grad

def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))	

@cuda.jit
def cuda_conv_forward(pic,f,pro):
	tx = cuda.threadIdx.x
	ty = cuda.threadIdx.y

	f_num = cuda.blockIdx.x

	fil = f[f_num]

	for i in range(3):
		for j in range(3):
			pro[tx, ty, f_num] += pic[tx+i, ty+j, f_num] * fil[i, j, 0]

@cuda.jit
def cuda_conv_backward(x, f, pro_grad, dx, f_grad):
	tx = cuda.threadIdx.x
	ty = cuda.threadIdx.y

	f_num = cuda.blockIdx.x

	fil = f[f_num]
	for i in range(3):
		for j in range(3):
			tmp1 = f[f_num,i,j,0] * pro_grad[tx,ty,f_num]
			tmp2 = x[tx+i,ty+j,0] * pro_grad[tx,ty,f_num]
			cuda.atomic.add(dx, (tx+i,ty+j,0), tmp1)
			cuda.atomic.add(f_grad, (f_num,i,j,0), tmp2)

@cuda.jit
def cuda_pool_forward(o,p):
	tx = cuda.threadIdx.x
	ty = cuda.threadIdx.y

	d = cuda.blockIdx.x

	p[tx,ty,d] = max(o[2*tx,2*ty,d],o[2*tx+1,2*ty,d],o[2*tx,2*ty+1,d],o[2*tx+1,2*ty+1,d])

@cuda.jit
def cuda_pool_backward(p_grad, o_grad):
	tx = cuda.threadIdx.x
	ty = cuda.threadIdx.y

	d = cuda.blockIdx.x

	o_grad[tx,ty,d] = p_grad[int(tx/2),int(ty/2),d]

# Prepare data
data, labels, testdata, testlabels = getMnistData()

def Adam(dx, m, v):
	beta1 = 0.9
	beta2 = 0.999
	eps = 1e-8
	m = beta1*m + (1-beta1)*dx
	v = beta2*v + (1-beta2)*(dx**2)
	newWts = learning_rate * m / (np.sqrt(v) + eps)	
	return newWts, m, v

def newWts():
	f1 = np.random.randn(8,3,3,1) * np.sqrt(1.0/9)

	w1 = np.random.randn(neurons1, 1352) * np.sqrt(1.0/1352)
	b1 = np.zeros([neurons1,1])

	w2 = np.random.randn(neurons2, neurons1) * np.sqrt(1.0/neurons1)
	b2 = np.zeros([neurons2,1])

	w3 = np.random.randn(10, neurons2) * np.sqrt(1.0/neurons2)
	b3 = np.zeros([10,1])

	return f1,w1,b1,w2,b2,w3,b3

def loadWts():
	f1 = loadWeight('conv_weights/1.pkl')
	w1 = loadWeight('conv_weights/2.pkl')
	b1 = loadWeight('conv_weights/3.pkl')
	w2 = loadWeight('conv_weights/4.pkl')
	b2 = loadWeight('conv_weights/5.pkl')
	w3 = loadWeight('conv_weights/6.pkl')
	b3 = loadWeight('conv_weights/7.pkl')
	return f1,w1,b1,w2,b2,w3,b3

f1,w1,b1,w2,b2,w3,b3 = newWts()

mf1 = np.zeros(f1.shape)
vf1 = np.zeros(f1.shape)
mw1 = np.zeros(w1.shape)
vw1 = np.zeros(w1.shape)
mb1 = np.zeros(b1.shape)
vb1 = np.zeros(b1.shape)
mw2 = np.zeros(w2.shape)
vw2 = np.zeros(w2.shape)
mb2 = np.zeros(b2.shape)
vb2 = np.zeros(b2.shape)
mw3 = np.zeros(w3.shape)
vw3 = np.zeros(w3.shape)
mb3 = np.zeros(b3.shape)
vb3 = np.zeros(b3.shape)

for epoch_no in range(0,epochs):
	num_correct = 0
	start = time()
	for index in range(0,60000):
		o1 = np.zeros([26,26,8])
		pro1 = np.zeros([26,26,8])
		cuda_conv_forward[(8),(26,26)](data[index], f1, pro1)
		o1 = reLU(pro1)
		p1 = np.zeros([13,13,8])
		cuda_pool_forward[(8),(13,13)](o1,p1)
		pflat = p1.flatten().reshape(-1,1)
		l1 = sigmoid(np.matmul(w1,pflat) + b1)
		l2 = sigmoid(np.matmul(w2, l1) + b2)
		l3 = sigmoid(np.matmul(w3, l2) + b3)
		
		if np.argmax(l3) == np.argmax(labels[index]):
			num_correct += 1
		
		pic_loss = (labels[index] - l3)**2
		
		if (index+1)%100 == 0:
			print(str(epoch_no) + ": " + str(index+1) + ": " + str(pic_loss.sum()))
		
		l3_grad = 2 * (l3 - labels[index])
		sig3_grad = l3_grad * l3 * (1-l3)
		b3_grad = sig3_grad
		wx3_grad = sig3_grad
		w3_grad = np.matmul(wx3_grad, np.transpose(l2))
		l2_grad = np.matmul(np.transpose(w3), wx3_grad)
		sig2_grad = l2_grad * l2 * (1-l2)
		b2_grad = sig2_grad
		wx2_grad = sig2_grad
		w2_grad = np.matmul(wx2_grad, np.transpose(l1))
		l1_grad = np.matmul(np.transpose(w2), wx2_grad)
		sig1_grad = l1_grad * l1 * (1-l1)
		b1_grad = sig1_grad
		wx1_grad = sig1_grad
		w1_grad = np.matmul(wx1_grad, np.transpose(pflat))
		pflat_grad = np.matmul(np.transpose(w1), wx1_grad)
		p_grad = pflat_grad.reshape(13,13,8)
		o1_grad = np.zeros([26,26,8])
		cuda_pool_backward[(8),(26,26)](p_grad, o1_grad)
		pro1_grad = reLU_back(pro1,o1_grad)
		dx = np.zeros([28,28,1])
		f1_grad = np.zeros([8,3,3,1])
		cuda_conv_backward[(8),(26,26)](data[index], f1, pro1_grad, dx, f1_grad)	
		
		f1_,mf1,vf1 = Adam(f1_grad,mf1,vf1)
		f1 -= f1_
		w1_,mw1,vw1 = Adam(w1_grad,mw1,vw1)
		w1 -= w1_
		b1_,mb1,vb1 = Adam(b1_grad,mb1,vb1)
		b1 -= b1_
		w2_,mw2,vw2 = Adam(w2_grad,mw2,vw2)
		w2 -= w2_
		b2_,mb2,vb2 = Adam(b2_grad,mb2,vb2)
		b2 -= b2_
		w3_,mw3,vw3 = Adam(w3_grad,mw3,vw3)
		w3 -= w3_
		b3_,mb3,vb3 = Adam(b3_grad,mb3,vb3)
		b3 -= b3_		

	print(str(epoch_no) + ": " + 'Training Accuracy: ' + str(num_correct/600) + '%')
	print('Time taken: ' + str((time()-start)) + 's')
	wts = [f1,w1,b1,w2,b2,w3,b3]
	saveWeights(wts)

num_correct = 0
for index in range(0,10000):
	o1 = np.zeros([26,26,8])
	pro1 = np.zeros([26,26,8])
	cuda_conv_forward[(8),(26,26)](testdata[index], f1, pro1)
	o1 = reLU(pro1)
	p1 = np.zeros([13,13,8])
	cuda_pool_forward[(8),(13,13)](o1,p1)
	pflat = p1.flatten().reshape(-1,1)
	l1 = sigmoid(np.matmul(w1,pflat) + b1)
	l2 = sigmoid(np.matmul(w2, l1) + b2)
	l3 = sigmoid(np.matmul(w3, l2) + b3)
	if np.argmax(l3) == np.argmax(testlabels[index]):
		num_correct += 1
	if (index+1) % 100 == 0:
		print(str(index+1))

print("Test Accuracy: " + str(num_correct/100) + '%')
