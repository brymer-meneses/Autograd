use autograd::Variable;

#[test]
fn it_works() {
    let a = Variable{
        value: 2 as f64,
        grad: 0 as f64,
        requires_grad: false,
        depends_on: Vec::new(), 
    };

    let b = Variable{
        value: 3 as f64,
        grad: 0 as f64,
        requires_grad: false,
        depends_on: Vec::new(), 
    };


    let c = a.clone() + b;
    c.backwards();

    // println!("{}", a.grad);
    
    assert_eq!(1 as f64, a.grad)

}



