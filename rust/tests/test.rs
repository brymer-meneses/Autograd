use autograd::new; 
use autograd::add;
use autograd::mul;
// use autograd::mul;


#[test]
fn variable() {
    let a = new!(2, true);

    assert_eq!(a.value, 2.0);
    assert_eq!(a.requires_grad, true);
    assert_eq!(a.grad, 0.0);
    assert_eq!(a.parent_nodes, Vec::new());
}

#[test]
fn add_single_grad() {
    let a = new!(2, true);
    let b = new!(2, false);

    let c = add!(a, b);
    
    assert_eq!(c.value, 4.0);
    assert_eq!(c.grad, 0.0);
    assert_eq!(c.requires_grad, true);
    assert_eq!(c.parent_nodes, vec![(&a, 1.0)]);
}

#[test]
fn add_multiple_grad() {
    let a = new!(5, true);
    let b= new!(2, true);

    let c = add!(a, b);

    assert_eq!(c.value, 7.0);
    assert_eq!(c.grad, 0.0);
    assert_eq!(c.requires_grad, true);
    assert_eq!(c.parent_nodes, vec![(&a, 1.0), (&b, 1.0)])
}

#[test]
fn mul_single_grad() {
    let a = new!(2, true);
    let b = new!(2, false);

    let c = mul!(a, b);
    
    assert_eq!(c.value, 4.0);
    assert_eq!(c.grad, 0.0);
    assert_eq!(c.requires_grad, true);
    assert_eq!(c.parent_nodes, vec![(&a, b.value)]);
}

#[test]
fn mul_multiple_grad() {
    let a = new!(5, true);
    let b= new!(2, true);

    let c = mul!(a, b);

    assert_eq!(c.value, 10.0);
    assert_eq!(c.grad, 0.0);
    assert_eq!(c.requires_grad, true);
    assert_eq!(c.parent_nodes, vec![(&a, b.value), (&b, a.value)]);
}
