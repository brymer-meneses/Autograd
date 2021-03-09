module autograd 
include("./ops.jl")

function ensure_float(n::Any) 
    if typeof(n) != Float64
        # Parse value to be Float64
        try
            n = convert(Float64, n)
        catch error 
            @warn("You passed a non-numerical for the parameter value in
                instantiating Variable")
        end
    end 
    return n
end 

mutable struct Variable 
    
    value::Float64
    requires_grad::Bool
    parent_nodes::Any
    grad::Float64

    function Variable(value, requires_grad=False, parent_nodes=Any[], grad=0)
        value = ensure_float(value)
        
        return new(value, requires_grad, parent_nodes, grad)
    end

    function Base.:+(v1::Variable, v2::Variable) 
        return add(v1, v2)
    end 

    function Base.:*(v1::Variable, v2::Variable)
        return mul(v1, v2)
    end

end

struct Node 
    # Node stores the dependencies of a 
    # variable. This is essential during 
    # the backpropagation.

    variable::Variable
    grad::Float64 

    function Node(variable::Variable, grad::Any)

        grad = ensure_float(grad)
        return new(variable, grad)
    end 
end 

function add(v1::Variable, v2::Variable)
    requires_grad = v1.requires_grad || v2.requires_grad
    result = v1.value + v2.value 

    parent_nodes = Any[]
    if v1.requires_grad
        push!(parent_nodes, Node(v1, 1))
    end

    if v2.requires_grad
        push!(parent_nodes, Node(v2, 1))
    end 

    return Variable(result, requires_grad, parent_nodes)
end 
        
function mul(v1::Variable, v2::Variable)
    requires_grad = v1.requires_grad || v2.requires_grad
    result = v1.value * v2.value 

    parent_nodes = Any[]
    if v1.requires_grad
        push!(parent_nodes, Node(v1, v2.value))
    end

    if v2.requires_grad
        push!(parent_nodes, Node(v2, v1.value))
    end 

    return Variable(result, requires_grad, parent_nodes)
end 

function backward(v::Variable)

    function compute_grads(v::Variable, child_grad::Float64)
    
        for node in v.parent_nodes

            parent_var = node.variable 
            if parent_var.requires_grad
                parent_grad = node.grad * child_grad
                parent_var.grad += parent_grad
            else 
                continue
            end 

        
        compute_grads(parent_var, parent_grad)
        end

    end
    compute_grads(v, 1.)

end 

function zero_grad(v::Variable)
    v.grad = 0.
end

export backward, zero_grad
export Variable

end # Module 