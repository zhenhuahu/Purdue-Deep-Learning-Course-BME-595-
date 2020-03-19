from logic_gates import AND
from logic_gates import OR
from logic_gates import NOT
from logic_gates import XOR


print("------- AND -------")
And = AND()
print(And(True, True))
print(And(True, False))
print(And(False, True))
print(And(False, False))


print("------- OR -------")
Or = OR()
print(Or(True, True))
print(Or(True, False))
print(Or(False, True))
print(Or(False, False))


print("------- NOT -------")
Not = NOT()
print(Not(True))
print(Not(False))


print("------- XOR -------")
Xor = XOR()
print(Xor(True, True))
print(Xor(True, False))
print(Xor(False, True))
print(Xor(False, False))
