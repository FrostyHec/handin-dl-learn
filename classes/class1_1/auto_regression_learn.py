from abc import ABC, abstractmethod
from enum import Enum
from numbers import Number

from utils.ArgsUtils import AU


class AbstractOperator(ABC):
    @abstractmethod
    def calculate(self, children: list["Node"]) -> "Node":
        pass

    @abstractmethod
    def backward(self, children: list["Node"], grad: Number):
        pass


class AddOperator(AbstractOperator):
    def calculate(self, children: list["Node"]) -> "Node":
        x0, x1 = AU.getN(children, 2)
        return Node(value=x0.value + x1.value, children=children, operator=self)

    def backward(self, children: list["Node"], grad: Number):
        for e in  AU.getN(children,2):
            e:Node
            e.backward(grad)

class MulOperator(AbstractOperator):

    def calculate(self, children: list["Node"]) -> "Node":
        x0, x1 = AU.getN(children, 2)
        return Node(value=x0.value * x1.value, children=children, operator=self)

    def backward(self, children: list["Node"], grad: Number):
        x0, x1 = AU.getN(children, 2)
        x0:Node
        x1:Node
        x0.backward(x1.value*grad)
        x1.backward(x0.value*grad)


class OperatorTypes(Enum):
    ADD = AddOperator()
    MUL = MulOperator()


class Node:
    def __init__(self, value, children: list["Node"] = None, operator: AbstractOperator = None, require_grad=True):
        self.value = value
        self.children = children
        self.operator = operator
        self.grad = None
        self.require_grad = require_grad

    @classmethod
    def from_child(cls, nodes: list["Node"], operator_type: OperatorTypes) -> "Node":
        return operator_type.value.calculate(nodes)

    def backward(self, backward_grad=None):
        if not self.require_grad: # not require grad
            return
        # init & accumulate grad
        if backward_grad is None:  # root node
            self.grad = 1
        else:
            if self.grad is None:
                self.grad = 0
            self.grad += backward_grad
        if self.operator is None: # leaf node, terminate
            return
        self.operator.backward(self.children, self.grad)


class NodeWrapper:
    def __init__(self, node):
        self.node = node

    @property
    def value(self):
        return self.node.value

    @property
    def grad(self):
        return self.node.grad

    def __add__(self, other: "NodeWrapper"):
        return NodeWrapper(Node.from_child([self.node, other.node], OperatorTypes.ADD))

    def __mul__(self, other: "NodeWrapper"):
        return NodeWrapper(Node.from_child([self.node, other.node], OperatorTypes.MUL))

    def backward(self):
        self.node.backward()

class Var(NodeWrapper):
    def __init__(self, value):
        super().__init__(Node(value))

class Const(NodeWrapper):

    def __init__(self, value):
        super().__init__(Node(value,require_grad=False))


if __name__ == "__main__":
    c = Const(2)
    z = Var(3)
    x = Var(4)
    k = Var(5)
    g = Var(6)
    y = c*z*x+k*x+g

    y.backward()
    print(f"x.grad:{x.grad};z.grad:{z.grad};k.grad:{k.grad};g.grad:{g.grad}")
