import pandas as pd

NN = 10
iter = 15

for k in range(iter):
    for agent in range(NN):
        datas = {'cost function': [k+1], 'Agent': [agent]}
        datas = pd.DataFrame(data=datas)
        print(datas)