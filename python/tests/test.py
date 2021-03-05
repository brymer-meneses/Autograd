from autograd import Variable

a = Variable(2, requires_grad=True)
b = Variable(3, requires_grad=True)


c = a**(1/2)
