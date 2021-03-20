

#[allow(dead_code)]
#[derive(Debug, PartialEq)]
pub struct Variable<'a> {

    // Stores the numerical value
    pub value: f64,

    // TODO add comment
    pub requires_grad: bool,

    // Stores the parent node/s that is/are required for backpropagation
    pub parent_nodes: Vec<(&'a Variable<'a>, f64)>,

    // Will be assigned a value after backpropagation
    pub grad: f64,
}


#[allow(dead_code)]
pub fn _add<'a> (v1: &'a Variable, v2: &'a Variable) -> Variable<'a> {
    let value = v1.value + v2.value;
    let requires_grad = v1.requires_grad || v2.requires_grad;
    let mut parent_nodes = Vec::new();

    if v1.requires_grad {
        let grad = 1.0;
        parent_nodes.push((v1, grad));
    }

    if v2.requires_grad {
        let grad = 1.0;
        parent_nodes.push((v2, grad));
    }

    return Variable {
        value,
        requires_grad,
        parent_nodes,
        grad: 0.0,
    }
}

#[allow(dead_code)]
pub fn _mul<'a> (v1: &'a Variable, v2: &'a Variable) -> Variable<'a> {
    let value = v1.value * v2.value;
    let requires_grad = v1.requires_grad || v2.requires_grad;
    let mut parent_nodes = Vec::new();


    if v1.requires_grad {
        let grad = v2.value;
        parent_nodes.push((v1, grad));
    }

    if v2.requires_grad {
        let grad= v1.value;
        parent_nodes.push((v2, grad));
    }

    return Variable {
        value,
        requires_grad,
        parent_nodes,
        grad: 0.0,
    }
}

trait AutoGrad {
    fn backwards(&mut self); 
    // fn zero_grad(&mut self) -> Self;
}

impl<'a> Variable<'a> {
    // ! Not working
    fn backwards(mut self) {
        fn cal_gradients<'a>(v: &mut Variable<'a>, parent_grad: f64) {
            for (variable, grad) in &mut v.parent_nodes {

                if variable.requires_grad {
                    let _grad = grad.clone();
                    let next_var_grad = parent_grad * _grad;
                    variable.grad += next_var_grad;
                } else {
                    continue 
                }
            }
        }
        cal_gradients(&mut self, 1.0);
    }
}

#[macro_export]
macro_rules! add {
    ($v1:ident, $v2:ident) => {
        {
            let result = autograd::_add(&$v1, &$v2);
            result
        }
    };
}

#[macro_export]
macro_rules! mul {
    ($v1:ident, $v2:ident) => {
        {
            let result = autograd::_mul(&$v1, &$v2);
            result
        }
    };
}

#[macro_export]
macro_rules! new {
    ($value:expr, $requires_grad:expr) => {
        {
            let value = $value as f64;
            autograd::Variable {
                value, 
                requires_grad: $requires_grad, 
                grad: 0.0,
                parent_nodes: Vec::new(),
            }
        }
        
    };
}
