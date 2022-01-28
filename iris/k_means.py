#importing modules
import matplotlib.pyplot as plt
import pandas as pd
from numpy import inf
from sklearn.model_selection import train_test_split as tts
from scipy.spatial.distance import euclidean
#from itertools import product as prd
from matplotlib.animation import FuncAnimation as fa


#loading data, making test train split
df = pd.read_csv('iris.csv')
df_train, df_test = tts(df, test_size=0.3, train_size = 0.7) #defaultly, train is 0.75 and test is 0.25

X_train = df_train[df_train.columns[:-1]]
Y_train = df_train[df_train.columns[-1]]

X_test = df_test[df_test.columns[:-1]]
Y_test = df_test[df_test.columns[-1]]

k = 3
X_train_n = X_train.to_numpy()
l_X_train_n = list(X_train_n)
k_means = X_train_n[:k]

M = 20 #iterations for k _means


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
        yield k_means, color_code_l
        
fig, ax = plt.subplots()

colors = {0:'r', 1:'g', 2:'b'}

def plot(ax, X_train_n, k_means, color_code_l):
    #plots on sepal length vs petal length plane
    for point,color_code in zip(X_train_n, color_code_l):
        ax.scatter(point[0], point[3], c=colors[color_code])
        
        for point in k_means:
            ax.scatter(point[0], point[3], c='k')
            
    return ax

def init():
    for point in X_train_n:
        ax.scatter(point[0], point[3], c='y')

def animate(j):
    ax.cla()
    k_means, color_code = j
    ax_obj = plot(ax, X_train_n, k_means, color_code)
    return [ax_obj]

ani = fa(fig, animate, frames = list(gen_k_means(X_train_n)),init_func = init,
                        blit = False, repeat = False)


plt.show()







    
    
