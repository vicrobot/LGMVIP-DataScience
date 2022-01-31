#importing modules
import pandas as pd
from numpy import inf
import numpy as np
#from scipy.spatial.distance import euclidean
from itertools import product as prd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as fa

#loading data
df = pd.read_csv('iris.csv')

X = df[df.columns[:-1]]
Y = df[df.columns[-1]]

k = 3
X_n = X.to_numpy()
l_X_n = list(X_n)

retries_c = 5 #have to make this much retries to overcome from the loss from sticking in local minima

"""
The moment grouping is introduced, the inter vs intra fight gets existed.
This fight helps k-means to minimize when right means are obtained (hopefully)
"""

#k means optimizer
def gen_k_means(X_n):
    k_means = X_n[:k]
    M = 20 #iterations for k _means
    #iterating M times
    for _ in range(M):
        color_code_l = []
        ml = np.asarray(list(prd(X_n, k_means)))
        dists = np.linalg.norm(ml[:,0] - ml[:,1], axis=1)
        idxs = dists.reshape((-1,k)).argmin(axis=1)
        color_code_l = idxs
        #for chosen neta and their corresponding public, this below stuff makes new election and new means
        #are produced(intra mean)
        df2 = pd.DataFrame({'vals':l_X_n, 'neta':idxs})
        grouped = df2.groupby('neta')
        k_means = (grouped.sum()/grouped.count()).to_numpy().flatten()
        yield k_means, color_code_l, _

class anim:
    def __init__(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1,2, figsize=(20,10))
        self.scat1 = self.ax1.scatter(X_n[:,0],X_n[:,2], c='y')
        self.scat2 = self.ax2.scatter(X_n[:,1],X_n[:,3], c='y')
        self.colors = {0:'r', 1:'g', 2:'b'}
        self.ax1.set_xlabel('Iris sepal length')
        self.ax1.set_ylabel('Iris petal length')
        self.ax2.set_xlabel('Iris sepal width')
        self.ax2.set_ylabel('Iris petal width')
        
    def animate(self,j):
        k_means, color_code, iter_n = j
        self.k_means = k_means
        main_frame1 = X_n[:, [0,2]]
        main_frame2 = X_n[:, [1,3]]
        plt.suptitle(f'Frame: {iter_n+1}')
        color = [self.colors[i] for i in color_code]
        frame_2_1 = []
        frame_2_2 = []
        for point in k_means:
            frame_2_1.append([point[0], point[2]])
            frame_2_2.append([point[1], point[3]])
        n_mf1 = np.append(main_frame1, frame_2_1).reshape(-1, 2)
        n_mf2 = np.append(main_frame2, frame_2_2).reshape(-1, 2)
        
        self.scat1.set_facecolor(color+['k']*k) #sets colors 
        self.scat1.set_offsets(n_mf1)  #sets coords for points to plot
        
        self.scat2.set_facecolor(color+['k']*k) #sets colors 
        self.scat2.set_offsets(n_mf2)  #sets coords for points to plot

    def run(self,gen_k_means):
        ani = fa(self.fig, self.animate, frames = gen_k_means(X_n), interval = 500,
                        blit = False, repeat = False)

        plt.show()
        return

anim().run(gen_k_means)

    
    
