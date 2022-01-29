#importing modules
import pandas as pd
from numpy import inf
import numpy as np
from sklearn.model_selection import train_test_split as tts
from scipy.spatial.distance import euclidean
#from itertools import product as prd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as fa



#loading data, making test train split
df = pd.read_csv('iris.csv')
df_train, df_test = tts(df, test_size=0.2, train_size = 0.8) #defaultly, train is 0.75 and test is 0.25

X_train = df_train[df_train.columns[:-1]]
Y_train = df_train[df_train.columns[-1]]

X_test = df_test[df_test.columns[:-1]]
Y_test = df_test[df_test.columns[-1]]

k = 3
X_train_n = X_train.to_numpy()
l_X_train_n = list(X_train_n)
k_means = X_train_n[:k]

M = 20 #iterations for k _means
retries_c = 5


"""
The moment grouping is introduced, the inter vs intra fight gets existed.
This fight helps k-means to minimize when right means are obtained (hopefully)
"""

def gen_k_means(k_means):
    #iterating M times
    for _ in range(M):
        #X_train_runner stores corresponding neta of that value
        X_train_runner = []
        color_code_l = []
        for value in X_train_n:
            chosen_mean = k_means[0]
            dist = inf
            color_code = 0
            counter = 0
            for i in range(k):
                mean = k_means[i]
                new_dist = euclidean(mean, value)
                
                if dist > new_dist:
                    dist = new_dist
                    chosen_mean = mean
                    color_code = counter
                counter += 1
            color_code_l.append(color_code)
            X_train_runner.append(tuple(chosen_mean))

        #for chosen neta and their corresponding public, this below stuff makes new election and new means
        #are produced(intra mean)
        df2 = pd.DataFrame({'vals':l_X_train_n, 'neta':X_train_runner})
        grouped = df2.groupby('neta')
        k_means = (grouped.sum()/grouped.count()).to_numpy().flatten()
        yield k_means, color_code_l, _


class anim:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.scat = self.ax.scatter(X_train_n[:,0],X_train_n[:,2], c='y')
        self.colors = {0:'r', 1:'g', 2:'b'}
        self.ax.set_xlabel('Iris sepal length')
        self.ax.set_ylabel('Iris petal length')
        
    def animate(self,j):
        k_means, color_code, iter_n = j
        self.k_means = k_means
        main_frame = X_train_n[:, [0,2]]
        self.ax.set_title(f'Frame: {iter_n+1}')
        color = [self.colors[i] for i in color_code]
        frame_2 = []
        for point in k_means:
            frame_2.append([point[0], point[2]])
        n_mf = np.append(main_frame, frame_2).reshape(-1, 2)
        
        self.scat.set_facecolor(color+['k']*k) #sets colors 
        self.scat.set_offsets(n_mf)  #sets coords for points to plot

    def run(self,gen_k_means):
        ani = fa(self.fig, self.animate, frames = gen_k_means(X_train_n), interval = 500,
                        blit = False, repeat = False)

        plt.show()
        return 


anim().run(gen_k_means)

    
    
