// use std::ops;

struct Variable<'a> {
    pub value: f64, 
    requires_grad: bool,
    depends_on: Vec<(&'a Variable<'a> ,f64)>,
    pub grads: f64,

}

trait BasicOperations<'a> {
    fn add(v1: &'a Variable, v2: &'a Variable) -> &'a Variable<'a>;
    fn mul(v1: &'a Variable, v2: &'a Variable) -> &'a Variable<'a>;
}

trait TrigonometricFunctions {
    fn cos<'a> (v1: &'a Variable) -> &'a Variable<'a>;
    fn sin<'a> (v1: &'a Variable) -> &'a Variable<'a>;
}

impl<'a> BasicOperations for Variable<'a> {
    fn add(v1: &'a Variable, v2: &'a Variable) -> &'a Variable<'a> {

        let result = v1.value + v2.value;
        let requires_grad = v1.requires_grad || v2.requires_grad;
        let mut dependency = Vec::new();

        if v1.requires_grad {
            dependency.push((v1, 1 as f64));
        } 
        
        if v2.requires_grad {
            dependency.push((v2, 1 as f64));
        }

        return &Variable {
            value: result,
            requires_grad: requires_grad,
            depends_on: dependency,
            grads: 0 as f64,
        }
    }

    fn mul(v1: &'a Variable, v2: &'a Variable) -> &'a Variable<'a> {
        
        let result = v1.value * v2.value;
        let requires_grad = v1.requires_grad || v2.requires_grad;
        let mut dependency = Vec::new();

        if v1.requires_grad {
            dependency.push((v1, v2.value));
        } 
        
        if v2.requires_grad {
            dependency.push((v2, v1.value));
        }

        return &Variable {
            value: result,
            requires_grad: requires_grad,
            depends_on: dependency,
            grads: 0 as f64,
        }
    }

}