import os
import unittest

from ..autograd import Variable



class TestVariableGrad(unittest.TestCase):

    def test_add(self):
        a = Variable(2, requires_grad=True)
        b = Variable(3)

        c = a + b
        c.backwards()
        self.assertEqual(a.grads, 1)
        self.assertEqual(b.grads, 0)

    def test_mul(self):
        a = Variable(2, requires_grad=True)
        b = Variable(3)

        c = a * b
        c.backwards()
        self.assertEqual(a.grads, b.value)
        self.assertEqual(b.grads, 0)


if __name__ == "__main__":
    print(os.getcwd())

    
