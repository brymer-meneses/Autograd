use std::ops;
use std::any::Any;

pub struct Variable {
    pub value: f64,
    pub requires_grad: bool, 
    pub depends_on: Vec<(Variable, f64)>,
    pub grad: f64,
}

// #[derive(Default)]
impl Default for Variable {
    fn default() -> Self {
        Variable {
            value: 0 as f64,
            grad: 0 as f64,
            requires_grad: false,
            depends_on: Vec::new(),
        }

    }
}


impl Variable {
    pub fn backwards(self) {
    
        fn compute_gradients(variable: Variable, previous_grad: f64) {
            for dependency in variable.depends_on {
                let mut next_variable = dependency.0;
                let grad = dependency.1;

                let this_variable_grads = previous_grad * grad;
                next_variable.grad += this_variable_grads;

                compute_gradients(next_variable, this_variable_grads);


            }
        }

        compute_gradients(self, 1 as f64);
    }

    pub fn new(value: f64, requires_grad: bool) -> Variable {

        let new_variable = Variable {
            value: value, 
            grad: 0.,
            requires_grad: requires_grad, 
            depends_on: Vec::new(),
        };

        return new_variable

    }
}

impl ops::Add for Variable {
    type Output = Variable;

    fn add(self, v: Variable) -> Variable{
        let result = self.value + v.value;
        let requires_grad = self.requires_grad || v.requires_grad;
        let mut dependency = Vec::new();

        if self.requires_grad {
            dependency.push((self, 1 as f64));
        }

        if v.requires_grad {
            dependency.push((v, 1 as f64));
        }

        return Variable{
            value: result, 
            requires_grad: requires_grad,
            depends_on: dependency,
            grad: 0 as f64,
        }
    }
}

impl ops::Mul for Variable {
    type Output = Variable;

    fn mul(self, v: Variable) -> Variable{
        let lhs_value = self.value;
        let rhs_value = v.value;

        let result = lhs_value * rhs_value;
        let requires_grad = self.requires_grad || v.requires_grad;
        let mut dependency = Vec::new();

        if self.requires_grad {
            dependency.push((self, lhs_value));
        }

        if v.requires_grad {
            dependency.push((v, rhs_value));
        }

        return Variable{
            value: result, 
            requires_grad: requires_grad,
            depends_on: dependency,
            grad: 0 as f64,
        }
    }

}