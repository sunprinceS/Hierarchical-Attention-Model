import numpy as np
import matplotlib.pyplot as plt

#HAM(phrase) HAM(sentence) DMN MemNN TreeLSTM LSTM QC_sim choice_sim
conf_arr = [[0.50,0.23,0.43,0.73,0.44,0.31,0.77,0.92],
            [0.11,0.22,0.33,0.44,0.55,0.66,0.98,0.12],
            [0.14,0.52,0.43,0.41,0.85,0.16,0.28,0.62]]

norm_conf = []
for i in conf_arr:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
        tmp_arr.append(float(j)/float(a))
    norm_conf.append(tmp_arr)

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                interpolation='nearest')

width, height = 3,8

for x in xrange(width):
    for y in xrange(height):
        ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center')

#HAM(phrase) HAM(sentence) DMN MemNN TreeLSTM LSTM QC_sim choice_sim
cb = fig.colorbar(res)
plt.xticks(range(height), ['HAM(p)','HAM(s)','DMN','MemNN','TreeLSTM','LSTM','qc sim','choice_sim'],rotation=45)
plt.yticks(range(width), ['type1','type2','type3'])
plt.tight_layout()
plt.savefig('confusion_matrix.png', format='png')
