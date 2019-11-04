# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:21:00 2019

@author: Ravi Teja
"""

from pylab import *
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from time import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def load_data():
    output_dim = 10
    data = pd.read_csv("train.csv",index_col = "id")
    data = data.values
    train_y = data[:,-1]
    train_x = (data[:,0:-1]>127).astype(int)
    data = pd.read_csv("test.csv",index_col = "id")
    data = data.values
    test_y = data[:,-1]
    test_x = (data[:,0:-1]>127).astype(int)
    return train_x,train_y,test_x,test_y

def logistic(x,w,b):
   xw = np.dot(x, w)
   replicated_b = np.tile(b, (x.shape[0], 1))

   return 1.0 / (1 + np.exp(- xw - b))

def rbm(dataset, num_hidden, learn_rate, epochs, k,batchsize):

   num_visible = dataset.shape[1]
   num_examples = dataset.shape[0]

   print("Training RBM")

   start_time = time()

   batches = num_examples // batchsize

   w = 0.1 * np.random.randn(num_visible, num_hidden)
   a = np.zeros((1, num_visible))
   b = -4.0 * np.ones((1, num_hidden))

   w_inc = np.zeros((num_visible, num_hidden))
   a_inc = np.zeros((1, num_visible))
   b_inc = np.zeros((1, num_hidden))

   save_error = []
   img_arr = []
   for epoch in range(epochs):
      error = 0
      for batch in range(batches):
         v0_init = dataset[int(batch*batchsize):int((batch+1)*batchsize)]
         v0 = v0_init
         v1 = v0
         for i in range(k) :
             if k!= 0:
                 v0 = v1
             prob_h0 = logistic(v0, w, b)
             h0 = (prob_h0 > np.random.rand(batchsize, num_hidden)).astype(int)
             prob_v1 = logistic(h0, w.T, a)
             v1 = (prob_v1 > np.random.rand(batchsize,num_visible)).astype(int)
             prob_h1 = logistic(v1, w, b)

         vh0 = np.dot(v0.T, prob_h0)

         poshidact = np.sum(prob_h0, axis=0)
         posvisact = np.sum(v0, axis=0)
         if epoch%1 == 0 and batch == 0:
             img_arr.append(v1[0].reshape(28,28))

         vh1 = np.dot(v1.T, prob_h1)

         neghidact = np.sum(prob_h1, axis=0)
         negvisact = np.sum(v1, axis=0)


         m = 0.5 if epoch > 5 else 0.9

         w_inc = w_inc * m + (learn_rate/batchsize) * (vh0 - vh1)
         a_inc = a_inc * m + (learn_rate/batchsize) * (posvisact - negvisact)
         b_inc = b_inc * m + (learn_rate/batchsize) * (poshidact - neghidact)

         a += a_inc
         b += b_inc
         w += w_inc

         error += np.sum((v0_init - v1) ** 2)
      save_error.append(error/60000)
      print("Epoch = %s. Reconstruction error = %0.2f. Time elapsed (sec): %0.2f. lr= %0.7f"
            % (epoch + 1, error/60000, time() - start_time, learn_rate))

   print ("Training done.\nTotal training time (sec): %0.2f \n" % (time() - start_time))
   np.savetxt('loss_h_'+str(num_hidden)+'_k_'+str(k)+'.txt', save_error)
   return w, a, b

def sample_hidden(v0,w,b):
   num_hidden = w.shape[1]
   return (logistic(v0, w, b) > np.random.rand(1, num_hidden)).astype(int)

def reconstruct(v0, w, a, b):
   num_hidden = w.shape[1]
   prob_h0 = logistic(v0, w, b)
   h0 = prob_h0 > np.random.rand(1, num_hidden)
   return (logistic(h0, w.T, a) > np.random.rand(1, num_visible)).astype(int)

def test_mnist(dataset, labels, n_examples, num_hidden, epochs, learn_rate,k):
   images = dataset
   labels = labels

   w = np.load(r'./weights/w_v60000_h'+str(num_hidden)+'.npy')
   a = np.load(r'./weights/a_v60000_h'+str(num_hidden)+'.npy')
   b = np.load(r'./weights/b_v60000_h'+str(num_hidden)+'.npy')
   print("Saving weights")
   save_weights(w, a, b, "Output", n_examples, num_hidden)

   print("Generating and saving the reconstructed images")
   samples = 10000
   images, labels = dataset, labels
   visible_rep = np.zeros((samples, 784))
   visible_label = np.zeros((samples,1))

   hidden_rep = np.zeros((samples, num_hidden))
   hidden_label = np.zeros((samples,1))
   for i in range(samples):
      data = images[i]
      data1 = reconstruct(data, w, a, b)
      visible_rep[i] = data1
      visible_label[i] = labels[i]
      data2 = sample_hidden(data, w, b)
      hidden_rep[i] = data2
      hidden_label[i] = labels[i]

   np.savetxt('visible_representation_n_h_'+str(num_hidden)+'_k_'+str(k)+'.txt', visible_rep)
   np.savetxt('visible_representation_labels_n_h_'+str(num_hidden)+'_k_'+str(k)+'.txt', visible_label)

   np.savetxt('hidden_representation_nh_'+str(num_hidden)+'_k_'+str(k)+'.txt', hidden_rep)
   np.savetxt('hidden_representation_labels_nh_'+str(num_hidden)+'_k_'+str(k)+'.txt', hidden_label)



def save_weights(w, a, b, directory, n_examples, num_hidden):

   if not os.path.exists(directory):
      os.makedirs(directory)

   w_name = directory + os.sep + "w_v" + str(n_examples) + "_h" + str(num_hidden)
   np.save(w_name, w)
   a_name = directory + os.sep + "a_v" + str(n_examples) + "_h" + str(num_hidden)
   np.save(a_name, a)
   b_name = directory + os.sep + "b_v" + str(n_examples) + "_h" + str(num_hidden)
   np.save(b_name, b)


train_x,train_y,test_x,test_y = load_data()
num_hidden = 400
num_visible = 784
learn_rate = 0.9
epochs = 10
k = 1
batchsize = 500
w,a,b = rbm(train_x, num_hidden, learn_rate, epochs, k, batchsize)
directory = r'./weights'
save_weights(w, a, b, directory, 60000, num_hidden)
test_mnist(test_x, test_y, 10000, num_hidden, epochs, learn_rate,k)
X = np.loadtxt("hidden_representation_nh_"+str(num_hidden)+"_k_"+str(k)+".txt")
pca = PCA(0.95)#n_components=50)
pca.fit(X)
X = pca.transform(X)
Y = TSNE(n_components=2).fit_transform(X)

label = np.loadtxt("hidden_representation_labels_nh_"+str(num_hidden)+"_k_"+str(k)+".txt")
style = 'seaborn-darkgrid'
plt.style.use(style)
plt.figure(figsize=(12,10))
colors = ['#04FEFF','#042CFF','#FF0000','#A40CE8','#D3AE0D','#0D9D26','#89FFAD','#FF2FFC','#5D1BFF','#573614']
plt.scatter(Y[:, 0], Y[:, 1], c=label, s = 20, cmap=matplotlib.colors.ListedColormap(colors),alpha=0.8)
cb = plt.colorbar()
cb.set_ticks([0,1,2,3,4,5,6,7,8,9])
cb.set_ticklabels(np.arange(9))
plt.title("t-SNE representation of hidden units ")
plt.show()


