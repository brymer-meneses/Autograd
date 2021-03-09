include("../src/autograd.jl")

import .autograd: Variable, backward, zero_grad

a = Variable(2, true)
b = Variable(5, true)
c = a + b
d = a * b

# println("c = ", c.value)
# backward(c)
# println("∇a = ", a.grad )

# zero_grad(b)

# println("d = ", d.value)
# backward(d)
# println("∇b = ", b.grad)

e = c * d
backward(e)
           
println(a.grad)


