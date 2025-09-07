import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core import Value


a = Value(2.0)
b = Value(4.0)
c = a - b
print(c)