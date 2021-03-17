// use std::ops;

struct Variable<'a> {
    pub value: f64, 
    requires_grad: bool,
    depends_on: Vec<(&'a Variable<'a> ,f64)>,
    pub grads: f64,

}

trait BasicOperations {
    fn add(v1: &Variable, v2: &Variable) -> &Variable;
    fn mul(v1: &Variable, v2: &Variable) -> &Variable;
}

impl BasicOperations for Variable {
    fn add(v1: &Variable, v2: &Variable) -> &Variable {
        let result = v1.value + v2.value;
        let requires_grad = v1.requires_grad || v2.requires_grad;
        let mut dependency = Vec::new();

        if v1.requires_grad {
            dependency.append((v1, 1 as f64));
        }
    }
}