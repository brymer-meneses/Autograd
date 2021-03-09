include("../src/autograd.jl")

using .autograd: Variable, backwards


a = Variable(2., true)
b = Variable(3.)

c = a * b
backwards(c)

print(a.grad)
print(b.grad)