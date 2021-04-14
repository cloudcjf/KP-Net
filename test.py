from tqdm import tqdm
from time import sleep
from tester import a

data_loader = list(range(1000))
print(a)
for i, j in enumerate(tqdm(data_loader)):
    sleep(0.01)