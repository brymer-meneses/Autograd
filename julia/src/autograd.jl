module autograd 

include("./variable.jl")
include("./helper.jl")

import .variable: Variable, Node
import .helper: ensure_float


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

end # Module 