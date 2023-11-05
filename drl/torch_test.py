import socket
import torch

torch.set_num_interop_threads(1)
torch.set_num_threads(1)
print("cuda=%s on %s"%(
    torch.cuda.is_available(),
    socket.gethostname()))