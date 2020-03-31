import numpy as np


def nonlin(x, deriv=False):
    if(deriv == True):
        return (x*(1-x))

    return 1/(1+np.exp(-x))


def step(x, deriv=False):
	if(deriv == True):
	x[x < 0] = 0
	return x
	x[x < 0.5] = 0
	x[x >= 0.5] = 1
	return x


x = np.array([
             [0, 0, 0, 0, 1],
             [0, 0, 0, 1, 1],
             [0, 0, 1, 0, 1],
             [0, 0, 1, 1, 1],
             [0, 1, 0, 0, 1],
             [0, 1, 0, 1, 1],
             [0, 1, 1, 0, 1],
             [0, 1, 1, 1, 1],
             [1, 0, 0, 0, 1],
             [1, 0, 0, 1, 1],
             [1, 0, 1, 0, 1],
             [1, 0, 1, 1, 1],
             [1, 1, 0, 0, 1],
             [1, 1, 0, 1, 1],
             [1, 1, 1, 0, 1],
             [1, 1, 1, 1, 1]])

y = np.array([[0], [1], [1], [1], [1], [0], [0], [0],
              [0], [1], [1], [1], [0], [0], [0], [0]])


#seed

np.random.seed(1)

#synapses

syn0 = 2*np.random.random((5, 3)) - 1
syn1 = 2*np.random.random((3, 1)) - 1

#traning

for j in range(60000):

	#layers
	l0 = x
	l1 = nonlin(np.dot(l0, syn0))
	l2 = nonlin(np.dot(l1, syn1))

	#backpropagation
	l2_error = y - l2
	if (j % 1000) == 0:
            print('Error at iter ' + str(j) + ':' +
                  str(np.mean(np.abs(l2_error))))

	l2_delta = l2_error * nonlin(l2, deriv=True)
	l1_error = l2_delta.dot(syn1.T)
	l1_delta = l1_error * nonlin(l1, deriv=True)

	#update our synapses
	syn1 += l1.T.dot(l2_delta)
	syn0 += l0.T.dot(l1_delta)

print('Output after training')
l0 = x
l1 = step(np.dot(l0, syn0))
l2 = step(np.dot(l1, syn1))
print(syn0)
print(syn1)
print(l2)
