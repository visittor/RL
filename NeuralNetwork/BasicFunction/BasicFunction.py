# from ._BasicFunction import add, mm, max
from ._BasicFunction import Add, Substract, MatMul, Multiply, Divide
from ._BasicFunction import Exp, Log, Max, ReduceSum

add = Add.apply
substract = Substract.apply
mm = MatMul.apply
multiply = Multiply.apply
divide = Divide.apply
exp = Exp.apply
log = Log.apply
max = Max.apply
reduce_sum = ReduceSum.apply