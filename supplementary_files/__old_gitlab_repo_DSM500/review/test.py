import torch
import debugpy
print("Attaching debugger ...")
debugpy.listen(("0.0.0.0", 5678))
debugpy.wait_for_client()
print("Debugger attached!")

print(torch.__version__)

a = 0

for i in range(10):
    a += i

import debugpy
debugpy.breakpoint()

print(a)