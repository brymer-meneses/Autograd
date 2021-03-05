from collections import defaultdict


class Variable:
    VARIABLE_COUNT = 0
    grads = 0
    leaf_variables = []

    def __init__(self, value, requires_grad, depends_on=[], is_leaf=True):
        self.value = value
        self.depends_on = depends_on
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf

        if self.is_leaf:
            self._register_leaf_variable(self)

        return

    def __add__(self, v):
        result = self.value + v.value
        requires_grad = self.requires_grad or v.requires_grads

        dependency = []
        if self.requires_grad:
            dependency.append(
                (self, 1)
            )
        if v.requires_grad:
            dependency.append(
                (v, 1)
            )

        return Variable(result, requires_grad, depends_on=dependency, is_leaf=False)

    def __mul__(self, v):
        result = self.value + v.value
        requires_grad = self.requires_grad or v.requires_grads

        dependency = []
        if self.requires_grad:
            dependency.append(
                (self, v.value)
            )
        if v.requires_grad:
            dependency.append(
                (v, self.value)
            )

        return Variable(result, requires_grad, depends_on=dependency, is_leaf=False)

    def __pow__(self, num):
        result = self.value ** num
        requires_grad = self.requires_grad

        dependency = []
        grad = num * (self.value**(num-1))
        if self.requires_grad:
            dependency.append(
                (self, grad)
            )

        return Variable(result, requires_grad, depends_on=dependency, is_leaf=False)

    def backwards(self):

        def compute_gradients(variable, previous_grad):

            for next_variable, grad in variable.depends_on:
                # dV_(n)/dV_(n-1)
                this_variable_grads = previous_grad * grad
                next_variable.grads += this_variable_grads

                compute_gradients(next_variable, this_variable_grads)

        compute_gradients(self, previous_grad=1)

        return

    def _registerID(self):
        id = self._getID()
        self.id = id
        return

    @classmethod
    def _get_leaf_variables(cls):
        return cls.leaf_variables

    @classmethod
    def _getID(cls):
        id = cls.VARIABLE_COUNT
        cls.VARIABLE_COUNT += 1
        return id

    @classmethod
    def _register_leaf_variable(cls, variable):
        cls.leaf_variables.append(variable)
        return
