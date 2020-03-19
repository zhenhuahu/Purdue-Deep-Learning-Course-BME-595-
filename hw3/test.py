from logicGates import AND
from logicGates import OR
from logicGates import NOT
from logicGates import XOR


print("------- AND -------")
And = AND()
And.train()
print(And(True, True))
print(And(True, False))
print(And(False, True))
print(And(False, False))


print("------- OR -------")
Or = OR()
Or.train()
print(Or(True, True))
print(Or(True, False))
print(Or(False, True))
print(Or(False, False))


print("------- NOT -------")
Not = NOT()
Not.train()
print(Not(True))
print(Not(False))


print("------- XOR -------")
Xor = XOR()
Xor.train()
print(Xor(True, True))
print(Xor(True, False))
print(Xor(False, True))
print(Xor(False, False))
