import json
import matplotlib.pyplot as plt
import os
import numpy as np
import itertools

labels, fms = [], []
data = []
for result_file in os.listdir('../output/result/'):
    with open('../output/result/' + result_file) as fp:
        datum = json.load(fp)
    fp.close()

    # if datum['fmeasure'] < 0.3:
    #     continue

    if datum['c'] < 4:
        continue

    data.append(datum)

data = sorted(data, key=lambda x: (x['c'], x['gamma']))


for datum in data:
    labels.append('%.02f %.02f' % (datum['c'], datum['gamma']))
    fms.append(datum['fmeasure'])

print(max(fms))
plt.plot(labels, fms)
plt.yticks(np.arange(min(fms), max(fms), 0.0025))
plt.show()
