module autograd

function greet()
    println("Hello")
end

struct Variable

    value::Float64
    requires_grad::Bool
    depends_on
    grad::Float64


    function Variable(value::Float64, requires_grad=false, depends_on=[]) 
        return new(value, requires_grad, depends_on)
    end

    function Base.:+(v1::Variable, v2::Variable)
        return add(v1, v2)
    end

    function Base.:*(v1::Variable, v2::Variable)
        return mul(v1, v2)
    end

end


function add(v1::Variable, v2::Variable)

    result = v1.value + v2.value 
    requires_grad = v1.requires_grad || v2.requires_grad

    Dependency = []

    if v1.requires_grad
        grad = 1.
        append!(Dependency, [v1, grad])
    end


    if v2.requires_grad
        grad = 1.
        append!(Dependency, [v2, grad])
    end
  
    new_var = Variable(result, requires_grad, Dependency) 
    return new_var
    
end 

function mul(v1::Variable, v2::Variable)
    result = v1.value * v2.value 
    requires_grad  = v1.requires_grad || v2.requires_grad

    Dependency = []
    
    v1.requires_grad
        let grad = v2.value 
        append!(Dependency, [v1, grad])
    end 
    
    v2.requires_grad
        let grad = v1.value 
        append!(Dependency, [v2, grad])
    end 
    
    new_var = Variable(result, requires_grad, Dependency)
    return new_var
end

function backwards(v::Variable) 
    
    function compute_gradients(v::Variable, prev_grad::Float64)
        for dependency in v.depends_on
            next_variable, grad = dependency

            # dv
            #       -> The next backward variable
            #           V_{n-1}

            # next_variable_grads
            #       -> Derivative of this variable with respect 
            #       the next backward variable
            #       (\frac{dV_n}{dV_{n-1}})
                       
            dv = prev_grad * grad 
            next_variable.grad += dv
        end
        compute_gradients(next_variable, dv)
    end 

    compute_gradients(v, 1.)
end 




end # module
