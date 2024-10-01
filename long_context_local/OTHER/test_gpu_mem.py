import torch

def calc_mem(dev_id):
    t = torch.cuda.get_device_properties(dev_id).total_memory
    r = torch.cuda.memory_reserved(dev_id)
    a = torch.cuda.memory_allocated(dev_id)
    f = r-a  # free inside reserved

    return t, f

print(torch.cuda.is_available())

print(torch.cuda.device_count())

for dev_id in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_properties(dev_id))
    print(calc_mem(dev_id))