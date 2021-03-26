
use std::cell::RefCell;
use std::ops::{DerefMut, Deref}; 


struct Node<'a> {
    var: RefCell<&'a Variable<'a>>, 
    grad: f64, 
}

pub struct Variable<'a> {
    pub value: f64, 
    pub grad: f64, 
    requires_grad: bool, 
    depends_on: Vec<Node<'a>>,
}

impl<'a> Deref for Variable<'a> {
    type Target = Variable<'a>;

    fn deref(&self) -> &Self::Target {
        return self;
    }
}

impl<'a> DerefMut for Variable<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        return self;
    }
}

// Autograd 
impl<'a> Variable<'a> {
    pub fn backward(&mut self, grad: f64) {
        for node in &mut self.depends_on {
            // Compute Gradient of this Variable with
            // respect to its parent 
            let this_grad = grad * node.grad;
            self.grad += this_grad; 

            // Recurse 
            let mut parent_node = node.var.borrow_mut();
            parent_node.backward(this_grad);
        }
    }
}


// Operators 
impl<'a> Variable<'a> {
    pub fn add(v1: &'a Variable<'a>, v2: &'a Variable<'a>) -> Variable<'a> {
        let value = v1.value + v2.value;
        let requires_grad = v1.requires_grad || v2.requires_grad; 
        let mut depends_on = Vec::new();
        
        if v1.requires_grad {
            let node = Node {
               var: RefCell::from(v1),
               grad: 1.0,
            };
            depends_on.push(node);
        };

        if v2.requires_grad {
            let node = Node {
                var: RefCell::from(v2),
                grad: 1.0,
            }; 
            depends_on.push(node)
        }

        return Variable {
            value,
            requires_grad,
            depends_on,
            grad: 0.0,
        }
    }
}

// Constructor 
impl<'a> Variable<'a> {
    pub fn new(value: f64, requires_grad: bool) -> Variable<'a> {
        return Variable {
            value,
            requires_grad,
            depends_on: Vec::new(),
            grad: 0.0,
        }
    } 
}
