module variable 
include("./helper.jl")

import .helper: ensure_float


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

mutable struct Node 
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


export Variable, Node

end 