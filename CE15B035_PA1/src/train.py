
# coding: utf-8

# In[ ]:
import os
import argparse
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import math
from sklearn.metrics import accuracy_score
parser = argparse.ArgumentParser()

parser.add_argument("--lr", type=float)
parser.add_argument("--momentum", type=float)
parser.add_argument("--num_hidden", type=int)
parser.add_argument("--sizes", type=str)
parser.add_argument("--activation", type=str)
parser.add_argument("--loss", type=str)
parser.add_argument("--opt", type=str)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--epochs", type=int)
parser.add_argument("--anneal", type=str)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--expt_dir", type=str)
parser.add_argument("--train", type=str)
parser.add_argument("--val", type=str)
parser.add_argument("--test", type=str)
parser.add_argument("--pretrain", type=str)
parser.add_argument("--state", type=int)
parser.add_argument("--testing", type=str)

args = parser.parse_args()

alpha = args.lr
eta = args.momentum
n = args.num_hidden
actFunc = args.activation
sizes = args.sizes.split(',')
loss_sc = args.loss
opt = args.opt
batchsize = args.batch_size
epocs = args.epochs
anneal = args.anneal
save_dir = args.save_dir
expt_dir = args.expt_dir
trainFile = args.train
valFile = args.val
testFile = args.test
pretrain = args.pretrain
state = args.state
testing = args.testing

# trainFile = trainFile.decode('utf-8')
# valFile = valFile.decode('utf-8')
# testFile = testFile.decode('utf-8')

print(testFile)

# print(type(epocs))
print(anneal)

print(testing)
print(type(testing))


import numpy as np
import pandas as pd
import math

input_size = 200
output_size = 10
#actFunc = "sigmoid"

#print(len(actFunc))

#print(type(actFunc))
#sizes = 10*np.ones(n) # sizes is sizes of hidden layer
sizes=np.array(sizes) #output layer not mentioned 
sizes = sizes.astype(np.int64)
# n=sizes.shape[0]
#epocs = 20
#alpha = 0.01


# In[20]:


#reading data
data = pd.read_csv(trainFile)


# In[21]:


data.shape[0]


# In[22]:


data = data.sample(frac=1)

sub_data = data.head(data.shape[0]) #partial data
sub_data = sub_data.values
print(np.max(data.values[:,-1]))


# In[23]:


ids = sub_data[:,1]
print(ids.shape)

images = sub_data[:,1:-1]
print(images.shape)

labels = sub_data[:,-1]
print(labels.shape)

labels = labels.astype(np.int64)


# In[24]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(images)
images = scaler.transform(images)
labels = labels


# In[25]:


from sklearn.decomposition import PCA

pca = PCA(n_components=200)

pca.fit(images)

images1 = pca.transform(images)
input_size=200
labels1=labels.copy()

# In[26]:


# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(images)
# images = scaler.transform(images)


# In[27]:


images.shape


# In[28]:


# weights=[]
# for i in range(1, n+2):
#     weights.append("W%s" %i);

# print(weights)


# In[29]:


# bias=[]

# for i in range(1, n+2):
#     bias.append("b%s" %i);
    
# print(bias)

# pact=[]

# for i in range(1, n+2):
#     pact.append("a%s" %i);
    
# print(pact)

# act=[]

# for i in range(1, n+2):
#     act.append("h%s" %i);
    
# print(act)


# In[30]:


NN_dict = {}


# In[31]:


data1 = pd.read_csv(valFile)

sub_data1 = data1 #partial data
sub_data1 = sub_data1.values
print(sub_data1[0].shape)

ids = sub_data1[:,1]
print(ids.shape)

images2 = sub_data1[:,1:-1]
print(images2.shape)

labels2 = sub_data1[:,-1]
print(labels2.shape)

labels2 = labels2.astype(np.int64)


# data3 = pd.read_csv("test.csv")

# sub_data3 = data3 #partial data
# sub_data3 = sub_data3.values
# print(sub_data3[0].shape)

# ids3 = sub_data3[:,0]
# print("a",ids3.shape)

# images3 = sub_data3[:,1:]
# print(images3.shape)

print(testing)



import sklearn
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(images2)
images2 = scaler.transform(images2)

images2=pca.transform(images2)

if (pretrain=="false"):
    state = -1

import pickle
# save_dir = "../save_dir/best/"
def save_weights(list_of_weights, epoch):
      with open(save_dir +'/weights_{}.pkl'.format(epoch), 'wb') as f:
             pickle.dump(list_of_weights, f)

def load_weights(state):
      with open(save_dir +'/best/weights_{}.pkl'.format(state),'rb') as f:
              list_of_weights = pickle.load(f)
      return list_of_weights


#Initialization of weights
def initialize(NN_dict, sizes):
    n=sizes.shape[0]
    weights=[]
    for i in range(1, n+2):
        weights.append("W%s" %i);
    bias=[]

    for i in range(1, n+2):
        bias.append("b%s" %i);

    #print(bias)

    pact=[]

    for i in range(1, n+2):
        pact.append("a%s" %i);

    #print(pact)

    act=[]

    for i in range(1, n+2):
        act.append("h%s" %i);

    #print(act)
    
    NN_dict[weights[0]] = (np.random.random_sample((sizes[0],input_size))-0.5)*0.1 #*np.sqrt(2/((sizes[0]+input_size)))
    for i in range(1,n):
            NN_dict[weights[i]] = (np.random.random_sample((sizes[i],sizes[i-1]))-0.5)*0.1 #*np.sqrt(2/((sizes[i]+sizes[i-1])))
    NN_dict[weights[-1]] = (np.random.random_sample((output_size,sizes[-1]))-0.5)*0.1 #*np.sqrt(2/((output_size+sizes[-1])))
    for i in range(0,n):
        NN_dict[bias[i]] = ((np.random.random_sample(sizes[i]))-0.5)*0.01
    NN_dict[bias[-1]] = ((np.random.random_sample(output_size))-0.5)*0.01
    
    for i in range(0,n):
            NN_dict[pact[i]] = np.zeros(sizes[i])
    NN_dict[pact[-1]] = np.zeros(output_size)
    #print(NN_dict[pact[-1]].shape)    
    
    for i in range(0,n):
            NN_dict[act[i]] = np.zeros(sizes[i])
    NN_dict[act[-1]] = np.zeros(output_size)
    #print(NN_dict[act[-1]].shape)
    print("q",len(load_weights(state)))
    if pretrain=="true":
        print(load_weights(state).shape)
        for i in range(len(weights)):
            NN_dict[weights[i]] = np.array(load_weights(state)[i])
        for i in range(len(weights),len(weights)+len(bias)):
            #print(load_weights(state)[i])
            NN_dict[bias[i-len(weights)]] = np.array(load_weights(state)[i])
            

    
    return NN_dict, bias, weights, pact,act






def activation(x,func):
    if func=="sigmoid":
        return 1/(1 + np.exp(-x))
    elif func=="tanh":
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    elif func=="relu":
        x[x<0]=0
        return x
    
def output(x):
    return (np.exp(x))/np.sum(np.exp(x))

def gdash(x,func):
    if func=="sigmoid":
        return activation(x,func)*(1-activation(x,func))
    elif func=="tanh":
        return 1-activation(x,func)*activation(x,func)
    elif func=="relu":
        #print(x.shape)
        x[x<0]=0
        x[x>0]=1
        return x

# In[35]:


def predict(input_x):    
    NN_dict[pact[0]] = NN_dict[bias[0]]+np.matmul(NN_dict[weights[0]],input_x)
    #print(actFunc)
    NN_dict[act[0]] = activation(NN_dict[pact[0]],actFunc)
    #print(NN_dict[act[0]])
    for p in range(1,n):
        #print("c",NN_dict[act[p-1]].shape)
        NN_dict[pact[p]] = NN_dict[bias[p]]+np.matmul(NN_dict[weights[p]],NN_dict[act[p-1]])
        #print("c",NN_dict[act[k-1]].shape)
        NN_dict[act[p]] = activation(NN_dict[pact[p]],actFunc)

        #print("a",p)
    NN_dict[pact[n]] = NN_dict[bias[n]]+np.matmul(NN_dict[weights[n]],NN_dict[act[n-1]])
    NN_dict[act[n]] = output(NN_dict[pact[n]])
    
    return np.argmax(NN_dict[act[n]])


def gradientdescent2(epocs,images1,NN_dict,batchsize,alpha,actFunc,labels1, bias, weights, pact,act, n, images11,labels11):
    losse=[]
    lossv=[]
    prev_errorV = 0
    for j in range(0,epocs):
        gradWkVecSum = np.array([])
        gradbkVecSum = np.array([])
        
        for q in range(0,images1.shape[0],batchsize):
            for i in range(q,q+batchsize):
                input_x = images1[i]
                label = labels1[i]

                #Forward-Prop 

                NN_dict[pact[0]] = NN_dict[bias[0]]+np.matmul(NN_dict[weights[0]],input_x)
                NN_dict[act[0]] = activation(NN_dict[pact[0]],actFunc)

                for p in range(1,n):
                    NN_dict[pact[p]] = NN_dict[bias[p]]+np.matmul(NN_dict[weights[p]],NN_dict[act[p-1]])
                    NN_dict[act[p]] = activation(NN_dict[pact[p]],actFunc)
                    #dropout
                    random1 = np.random.binomial(1,0.9,NN_dict[act[p]].shape[0])
                    NN_dict[act[p]] = np.multiply(NN_dict[act[p]],random1)


                NN_dict[pact[n]] = NN_dict[bias[n]]+np.matmul(NN_dict[weights[n]],NN_dict[act[n-1]])
                NN_dict[act[n]] = output(NN_dict[pact[n]])

                #Backward-Prop

                gradWkVec = []
                gradbkVec = []
                if(loss_sc=="ce"):
                    gradak = -(np.eye(1,output_size,labels1[i]).T[:,0] - NN_dict[act[-1]])
                if(loss_sc=="sq"):
                    gradak = -NN_dict[act[-1]][label]*(((np.eye(1,output_size,labels1[i]).T[:,0]- NN_dict[act[-1]]))**2)
                for k in range(n, 0, -1):
                    # Computing gradient w.r.t. parameters;
                    gradWk = np.matmul(gradak[np.newaxis].T,NN_dict[act[k-1]][np.newaxis]) #+0.01*NN_dict[weights[k]]  
                    gradbk = gradak.copy()

                    gradWkVec.append(gradWk)
                    gradbkVec.append(gradbk)

                    # Computing gradient w.r.t. layer below;
                    gradhkm1 = np.matmul(np.transpose(NN_dict[weights[k]]),gradak)

                    # Computing gradient w.r.t. layer below (pre-activation);
                    gradak = np.multiply(gradhkm1, gdash(NN_dict[pact[k-1]],actFunc))

                gradWk = np.matmul(gradak[np.newaxis].T,input_x[np.newaxis])    
                gradbk = gradak.copy()


                gradWkVec.append(gradWk)
                gradbkVec.append(gradbk)

                if i is 0:
                    gradWkVecSum = gradWkVec.copy()
                    gradbkVecSum = gradbkVec.copy()
                else:
                    for m in range(0,len(gradWkVecSum)):
                        gradWkVecSum[m] = gradWkVecSum[m] + gradWkVec[m]
                        gradbkVecSum[m] = gradbkVecSum[m] + gradbkVec[m]           
                
            #Update rule

            for l in range(0,n):
                NN_dict[weights[l]] = NN_dict[weights[l]] - alpha*gradWkVecSum[n-l]
                NN_dict[bias[l]] = NN_dict[bias[l]] - alpha*gradbkVecSum[n-l]


        #Loss-calc

        loss = 0

        for t in range(0,images1.shape[0]):    
            input_x = images1[t]
            label = labels1[t] 
            NN_dict[pact[0]] = NN_dict[bias[0]]+np.matmul(NN_dict[weights[0]],input_x)

            NN_dict[act[0]] = activation(NN_dict[pact[0]],actFunc)

            for k in range(1,n):
                NN_dict[pact[k]] = NN_dict[bias[k]]+np.matmul(NN_dict[weights[k]],NN_dict[act[k-1]])
                NN_dict[act[k]] = activation(NN_dict[pact[k]],actFunc)

            NN_dict[pact[-1]] = NN_dict[bias[-1]]+np.matmul(NN_dict[weights[-1]],NN_dict[act[-2]])
            NN_dict[act[-1]] = output(NN_dict[pact[-1]])
            if(loss_sc=="ce"):
                loss = loss-math.log(NN_dict[act[-1]][label])
            if(loss_sc=="sq"):
                loss = loss+0.5*np.sum(((np.eye(1,output_size,label).T[:,0] -NN_dict[act[-1]])**2))

                
        loss=loss/images1.shape[0]
        print(loss)
        losse.append(loss)

        loss1 = 0

        for t in range(0,images11.shape[0]):    
            input_x = images11[t]
            label = labels11[t] 
            NN_dict[pact[0]] = NN_dict[bias[0]]+np.matmul(NN_dict[weights[0]],input_x)

            NN_dict[act[0]] = activation(NN_dict[pact[0]],actFunc)

            for k in range(1,n):
                NN_dict[pact[k]] = NN_dict[bias[k]]+np.matmul(NN_dict[weights[k]],NN_dict[act[k-1]])
                NN_dict[act[k]] = activation(NN_dict[pact[k]],actFunc)

            NN_dict[pact[-1]] = NN_dict[bias[-1]]+np.matmul(NN_dict[weights[-1]],NN_dict[act[-2]])
            NN_dict[act[-1]] = output(NN_dict[pact[-1]])

            #loss1 = loss1-math.log(NN_dict[act[-1]][label])
            if(loss_sc=="ce"):
                loss1 = loss1-math.log(NN_dict[act[-1]][label])
            if(loss_sc=="sq"):
                loss1 = loss1+0.5*np.sum(((np.eye(1,output_size,label).T[:,0] -NN_dict[act[-1]])**2))
        loss1=loss1/images2.shape[0]
        #print(images1.shape[0])
        print(loss1)
        lossv.append(loss1)
        
        ypre1=[]
        for i in range(images1.shape[0]):
            ypre1.append(predict(images1[i]))
        from sklearn.metrics import accuracy_score
        errorT = 1-accuracy_score(labels1, ypre1)
        
        ypre2=[]
        for i in range(images11.shape[0]):
            ypre2.append(predict(images11[i]))
        from sklearn.metrics import accuracy_score
        errorV = 1-accuracy_score(labels11, ypre2)
        print(errorV)
        
        if j is 0:
            with open(expt_dir+"/log_train.txt", "w") as myfile:
                myfile.write("Epoch: %s, Step %s, Loss: %s, Error: %s, lr: %s \n" %(str(j), str(q), str(loss), str(errorT), str(alpha)))

            with open(expt_dir+"/log_val.txt", "w") as myfile:
                myfile.write("Epoch: %s, Step %s, Loss: %s, Error: %s, lr: %s\n" %(str(j), str(q), str(loss1), str(errorV), str(alpha)))
        else:
            with open(expt_dir+"/log_train.txt", "a") as myfile:
                myfile.write("Epoch: %s, Step %s, Loss: %s, Error: %s, lr: %s \n" %(str(j), str(q), str(loss), str(errorT), str(alpha)))

            with open(expt_dir+"/log_val.txt", "a") as myfile:
                myfile.write("Epoch: %s, Step %s, Loss: %s, Error: %s, lr: %s\n" %(str(j), str(q), str(loss1), str(errorV), str(alpha)))
    
        wt = []
        bt = []

        for u in range(len(weights)):
            wt.append(NN_dict[weights[u]])

        for u in range(len(bias)):
            wt.append(NN_dict[bias[u]])


        save_weights(np.array(wt),j+state+1)
        #save_bias(np.array(bt),j+state+1)
            
    if (anneal=="true"):
        if j>0:
            if errorV > prev_errorV:
                alpha = 0.5*alpha
        
    prev_errorV = errorV
    
    print("alpha",alpha)
    return losse , lossv


#n = sizes.shape[0]

NN_dict, bias, weights, pact,act = initialize(NN_dict,sizes)
# print(NN_dict)
# print(sizes)
# lossadam_t, lossadam_v , NN_dict = gradientdescent2(epocs,images1,NN_dict,batchsize,alpha,actFunc,labels1, bias, weights, pact,act, n,images2, labels2)

# loss_t.append(lossadam_t)
# loss_v.append(lossadam_v)


# In[39]:


#Momentum bases GD batch wise

# In[42]:


def momentum2(epocs,images1,NN_dict,batchsize,alpha,actFunc,labels1, bias, weights, pact,act, n, eta, images11, labels11):
    losse=[]
    lossv=[]
    prev_errorV = 0
    sumgradW=[]
    sumgradb=[]
    for j in range(0,epocs):
        gradWkVecSum = np.array([])
        gradbkVecSum = np.array([])

        for q in range(0,images1.shape[0],batchsize):
            for i in range(q,q+batchsize):
                input_x = images1[i]
                label = labels1[i]

                #Forward-Prop 

                NN_dict[pact[0]] = NN_dict[bias[0]]+np.matmul(NN_dict[weights[0]],input_x)
                NN_dict[act[0]] = activation(NN_dict[pact[0]],actFunc)

                for p in range(1,n):
                    NN_dict[pact[p]] = NN_dict[bias[p]]+np.matmul(NN_dict[weights[p]],NN_dict[act[p-1]])
                    NN_dict[act[p]] = activation(NN_dict[pact[p]],actFunc)
                    #dropout
                    random1 = np.random.binomial(1,0.9,NN_dict[act[p]].shape[0])
                    NN_dict[act[p]] = np.multiply(NN_dict[act[p]],random1)


                NN_dict[pact[n]] = NN_dict[bias[n]]+np.matmul(NN_dict[weights[n]],NN_dict[act[n-1]])
                NN_dict[act[n]] = output(NN_dict[pact[n]])

                #Backward-Prop

                gradWkVec = []
                gradbkVec = []

                if(loss_sc=="ce"):
                    gradak = -(np.eye(1,output_size,labels1[i]).T[:,0] - NN_dict[act[-1]])
                if(loss_sc=="sq"):
                    gradak = -NN_dict[act[-1]][label]*(((np.eye(1,output_size,labels1[i]).T[:,0]- NN_dict[act[-1]]))**2)

                for k in range(n, 0, -1):
                    # Computing gradient w.r.t. parameters;
                    gradWk = np.matmul(gradak[np.newaxis].T,NN_dict[act[k-1]][np.newaxis])    
                    gradbk = gradak.copy()

                    gradWkVec.append(gradWk)
                    gradbkVec.append(gradbk)

                    # Computing gradient w.r.t. layer below;
                    gradhkm1 = np.matmul(np.transpose(NN_dict[weights[k]]),gradak)

                    # Computing gradient w.r.t. layer below (pre-activation);
                    gradak = np.multiply(gradhkm1, gdash(NN_dict[pact[k-1]],actFunc))

                gradWk = np.matmul(gradak[np.newaxis].T,input_x[np.newaxis])    
                gradbk = gradak.copy()

                gradWkVec.append(gradWk)
                gradbkVec.append(gradbk)

                if i is 0:
                    gradWkVecSum = gradWkVec.copy()
                    gradbkVecSum = gradbkVec.copy()
                else:
                    for m in range(0,len(gradWkVecSum)):
                        gradWkVecSum[m] = gradWkVecSum[m] + gradWkVec[m]
                        gradbkVecSum[m] = gradbkVecSum[m] + gradbkVec[m]


            #Update rule
            if(j is 0):
                if( q is 0):
                    sumgradW , sumgradb = gradWkVecSum.copy() , gradbkVecSum.copy()
            else:
                for m in range(0,len(gradWkVecSum)):
                    sumgradW[m] = gradWkVecSum[m] + eta*sumgradW[m]
                    sumgradb[m] = gradbkVecSum[m] + eta*sumgradb[m]

            for l in range(0,n):
                #print(l)
                NN_dict[weights[l]] = NN_dict[weights[l]] - alpha*sumgradW[n-l]
                NN_dict[bias[l]] = NN_dict[bias[l]] - alpha*sumgradb[n-l]



        #Loss-calc

        loss = 0

        for t in range(0,images1.shape[0]):    
            input_x = images1[t]
            label = labels1[t] 
            NN_dict[pact[0]] = NN_dict[bias[0]]+np.matmul(NN_dict[weights[0]],input_x)

            NN_dict[act[0]] = activation(NN_dict[pact[0]],actFunc)

            for k in range(1,n):
                NN_dict[pact[k]] = NN_dict[bias[k]]+np.matmul(NN_dict[weights[k]],NN_dict[act[k-1]])
                NN_dict[act[k]] = activation(NN_dict[pact[k]],actFunc)

            NN_dict[pact[-1]] = NN_dict[bias[-1]]+np.matmul(NN_dict[weights[-1]],NN_dict[act[-2]])
            NN_dict[act[-1]] = output(NN_dict[pact[-1]])

            if(loss_sc=="ce"):
                loss = loss-math.log(NN_dict[act[-1]][label])
            if(loss_sc=="sq"):
                loss = loss+0.5*np.sum(((np.eye(1,output_size,label).T[:,0] -NN_dict[act[-1]])**2))
        loss = loss/images1.shape[0]
        print(loss)
        losse.append(loss)

        loss1 = 0

        for t in range(0,images11.shape[0]):    
            input_x = images11[t]
            label = labels11[t] 
            NN_dict[pact[0]] = NN_dict[bias[0]]+np.matmul(NN_dict[weights[0]],input_x)

            NN_dict[act[0]] = activation(NN_dict[pact[0]],actFunc)

            for k in range(1,n):
                NN_dict[pact[k]] = NN_dict[bias[k]]+np.matmul(NN_dict[weights[k]],NN_dict[act[k-1]])
                NN_dict[act[k]] = activation(NN_dict[pact[k]],actFunc)

            NN_dict[pact[-1]] = NN_dict[bias[-1]]+np.matmul(NN_dict[weights[-1]],NN_dict[act[-2]])
            NN_dict[act[-1]] = output(NN_dict[pact[-1]])

            if(loss_sc=="ce"):
                loss1 = loss1-math.log(NN_dict[act[-1]][label])
            if(loss_sc=="sq"):
                loss1 = loss1+0.5*np.sum(((np.eye(1,output_size,label).T[:,0] -NN_dict[act[-1]])**2))
        loss1=loss1/images2.shape[0]
        #print(images1.shape[0])
        print(loss1)
        lossv.append(loss1)
        
        ypre1=[]
        for i in range(images1.shape[0]):
            ypre1.append(predict(images1[i]))
        from sklearn.metrics import accuracy_score
        errorT = 1-accuracy_score(labels1, ypre1)
        
        ypre2=[]
        for i in range(images11.shape[0]):
            ypre2.append(predict(images11[i]))
        from sklearn.metrics import accuracy_score
        errorV = 1-accuracy_score(labels11, ypre2)
        
        if j is 0:
            with open(expt_dir+"/log_train.txt", "w") as myfile:
                myfile.write("Epoch: %s, Step %s, Loss: %s, Error: %s, lr: %s \n" %(str(j), str(q), str(loss), str(errorT), str(alpha)))

            with open(expt_dir+"/log_val.txt", "w") as myfile:
                myfile.write("Epoch: %s, Step %s, Loss: %s, Error: %s, lr: %s\n" %(str(j), str(q), str(loss1), str(errorV), str(alpha)))
        else:
            with open(expt_dir+"/log_train.txt", "a") as myfile:
                myfile.write("Epoch: %s, Step %s, Loss: %s, Error: %s, lr: %s \n" %(str(j), str(q), str(loss), str(errorT), str(alpha)))

            with open(expt_dir+"/log_val.txt", "a") as myfile:
                myfile.write("Epoch: %s, Step %s, Loss: %s, Error: %s, lr: %s\n" %(str(j), str(q), str(loss1), str(errorV), str(alpha)))
            
        wt = []
        bt = []

        for u in range(len(weights)):
            wt.append(NN_dict[weights[u]])

        for u in range(len(bias)):
            wt.append(NN_dict[bias[u]])


        save_weights(np.array(wt),j+state+1)
        #save_bias(np.array(bt),j+state+1)
            
    if (anneal=="true"):
        if j>0:
            if errorV > prev_errorV:
                alpha = 0.5*alpha
        
    prev_errorV = errorV
    
    print("alpha",alpha)
    return losse , lossv


# actFunc='sigmoid'

# n = sizes.shape[0]

# NN_dict, bias, weights, pact,act = initialize(NN_dict,sizes)

# eps=0.9

# lossadam_t, lossadam_v , NN_dict = momentum2(epocs,images1,NN_dict,batchsize,alpha,actFunc,labels1, bias, weights, pact,act, n,eps,images2, labels2)

# loss_t.append(lossadam_t)
# loss_v.append(lossadam_v)
# In[43]:


# epocs=10
# batchsize=100
# eta=0.9
# alpha=0.01
# #NN_dict=initialize(NN_dict)
# loss_moment1 , NN_dict= momentum2(epocs,images[20000:30000] , NN_dict,batchsize,eta,alpha)


# ## Nag GD batchwise

# In[44]:


def nag2(epocs,images1,NN_dict,batchsize,alpha,actFunc,labels1, bias, weights, pact,act, n,eta,images11, labels11):
    losse=[]
    lossv=[]
    prev_errorV = 0
    sumgradW=[]
    sumgradb=[]
    for j in range(0,epocs):
        gradWkVecSum = np.array([])
        gradbkVecSum = np.array([])

        for q in range(0,images1.shape[0],batchsize):
            for i in range(q,q+batchsize):
                input_x = images1[i]
                label = labels1[i]

                #Forward-Prop 

                NN_dict[pact[0]] = NN_dict[bias[0]]+np.matmul(NN_dict[weights[0]],input_x)
                NN_dict[act[0]] = activation(NN_dict[pact[0]],actFunc)

                for p in range(1,n):
                    NN_dict[pact[p]] = NN_dict[bias[p]]+np.matmul(NN_dict[weights[p]],NN_dict[act[p-1]])
                    NN_dict[act[p]] = activation(NN_dict[pact[p]],actFunc)

                NN_dict[pact[n]] = NN_dict[bias[n]]+np.matmul(NN_dict[weights[n]],NN_dict[act[n-1]])
                NN_dict[act[n]] = output(NN_dict[pact[n]])

                #Backward-Prop

                gradWkVec = []
                gradbkVec = []

                if(loss_sc=="ce"):
                    gradak = -(np.eye(1,output_size,labels1[i]).T[:,0] - NN_dict[act[-1]])
                if(loss_sc=="sq"):
                    gradak = -NN_dict[act[-1]][label]*(((np.eye(1,output_size,labels1[i]).T[:,0]- NN_dict[act[-1]]))**2)

                for k in range(n, 0, -1):
                    # Computing gradient w.r.t. parameters;
                    gradWk = np.matmul(gradak[np.newaxis].T,NN_dict[act[k-1]][np.newaxis])    
                    gradbk = gradak.copy()

                    gradWkVec.append(gradWk)
                    gradbkVec.append(gradbk)

                    # Computing gradient w.r.t. layer below;
                    gradhkm1 = np.matmul(np.transpose(NN_dict[weights[k]]),gradak)

                    # Computing gradient w.r.t. layer below (pre-activation);
                    gradak = np.multiply(gradhkm1, gdash(NN_dict[pact[k-1]],actFunc))

                gradWk = np.matmul(gradak[np.newaxis].T,input_x[np.newaxis])    
                gradbk = gradak.copy()

                gradWkVec.append(gradWk)
                gradbkVec.append(gradbk)

                if i is 0:
                    gradWkVecSum = gradWkVec.copy()
                    gradbkVecSum = gradbkVec.copy()
                else:
                    for m in range(0,len(gradWkVecSum)):
                        gradWkVecSum[m] = gradWkVecSum[m] + gradWkVec[m]


            #Update rule
            if(j is 0):
                if q is 0:
                    sumgradW , sumgradb = gradWkVecSum.copy() , gradbkVecSum.copy()
            else:
                for m in range(0,len(gradWkVecSum)):
                    sumgradW[m] =  eta*sumgradW[m]
                    sumgradb[m] =  eta*sumgradb[m]

            for l in range(0,n):
                NN_dict[weights[l]] = NN_dict[weights[l]] - alpha*sumgradW[n-l]
                NN_dict[bias[l]] = NN_dict[bias[l]] - alpha*sumgradb[n-l]


            for i in range(q,q+batchsize):
                input_x = images1[i]
                label = labels1[i]

                #Forward-Prop 

                NN_dict[pact[0]] = NN_dict[bias[0]]+np.matmul(NN_dict[weights[0]],input_x)
                NN_dict[act[0]] = activation(NN_dict[pact[0]],actFunc)

                for p in range(1,n):
                    NN_dict[pact[p]] = NN_dict[bias[p]]+np.matmul(NN_dict[weights[p]],NN_dict[act[p-1]])
                    NN_dict[act[p]] = activation(NN_dict[pact[p]],actFunc)

                NN_dict[pact[n]] = NN_dict[bias[n]]+np.matmul(NN_dict[weights[n]],NN_dict[act[n-1]])
                NN_dict[act[n]] = output(NN_dict[pact[n]])

                #Backward-Prop

                gradWkVec = []
                gradbkVec = []

                if(loss_sc=="ce"):
                    gradak = -(np.eye(1,output_size,labels1[i]).T[:,0] - NN_dict[act[-1]])
                if(loss_sc=="sq"):
                    gradak = -NN_dict[act[-1]][label]*(((np.eye(1,output_size,labels1[i]).T[:,0]- NN_dict[act[-1]]))**2)

                for k in range(n, 0, -1):
                    # Computing gradient w.r.t. parameters;
                    gradWk = np.matmul(gradak[np.newaxis].T,NN_dict[act[k-1]][np.newaxis])    
                    gradbk = gradak.copy()

                    gradWkVec.append(gradWk)
                    gradbkVec.append(gradbk)

                    # Computing gradient w.r.t. layer below;
                    gradhkm1 = np.matmul(np.transpose(NN_dict[weights[k]]),gradak)

                    # Computing gradient w.r.t. layer below (pre-activation);
                    gradak = np.multiply(gradhkm1, gdash(NN_dict[pact[k-1]],actFunc))

                gradWk = np.matmul(gradak[np.newaxis].T,input_x[np.newaxis])    
                gradbk = gradak.copy()

                gradWkVec.append(gradWk)
                gradbkVec.append(gradbk)

                if i is 0:
                    gradWkVecSum = gradWkVec.copy()
                    gradbkVecSum = gradbkVec.copy()
                else:
                    for m in range(0,len(gradWkVecSum)):
                        gradWkVecSum[m] = gradWkVecSum[m] + gradWkVec[m]


            #Update rule
            for m in range(0,len(gradWkVecSum)):
                sumgradW[m] = gradWkVecSum[m] + eta*sumgradW[m]
                sumgradb[m] = gradbkVecSum[m] + eta*sumgradb[m]

            for l in range(0,n):
                #print(l)
                NN_dict[weights[l]] = NN_dict[weights[l]] - alpha*sumgradW[n-l]
                NN_dict[bias[l]] = NN_dict[bias[l]] - alpha*sumgradb[n-l]

        #Loss-calc

        loss = 0

        for t in range(0,images1.shape[0]):    
            input_x = images1[t]
            label = labels1[t] 
            NN_dict[pact[0]] = NN_dict[bias[0]]+np.matmul(NN_dict[weights[0]],input_x)

            NN_dict[act[0]] = activation(NN_dict[pact[0]],actFunc)

            for k in range(1,n):
                NN_dict[pact[k]] = NN_dict[bias[k]]+np.matmul(NN_dict[weights[k]],NN_dict[act[k-1]])
                NN_dict[act[k]] = activation(NN_dict[pact[k]],actFunc)

            NN_dict[pact[-1]] = NN_dict[bias[-1]]+np.matmul(NN_dict[weights[-1]],NN_dict[act[-2]])
            NN_dict[act[-1]] = output(NN_dict[pact[-1]])

            if(loss_sc=="ce"):
                loss = loss-math.log(NN_dict[act[-1]][label])
            if(loss_sc=="sq"):
                loss = loss+0.5*np.sum(((np.eye(1,output_size,label).T[:,0] -NN_dict[act[-1]])**2))
        loss = loss/images1.shape[0]
        print(loss)
        losse.append(loss)

        loss1 = 0

        for t in range(0,images11.shape[0]):    
            input_x = images11[t]
            label = labels11[t] 
            NN_dict[pact[0]] = NN_dict[bias[0]]+np.matmul(NN_dict[weights[0]],input_x)

            NN_dict[act[0]] = activation(NN_dict[pact[0]],actFunc)

            for k in range(1,n):
                NN_dict[pact[k]] = NN_dict[bias[k]]+np.matmul(NN_dict[weights[k]],NN_dict[act[k-1]])
                NN_dict[act[k]] = activation(NN_dict[pact[k]],actFunc)

            NN_dict[pact[-1]] = NN_dict[bias[-1]]+np.matmul(NN_dict[weights[-1]],NN_dict[act[-2]])
            NN_dict[act[-1]] = output(NN_dict[pact[-1]])

            if(loss_sc=="ce"):
                loss1 = loss1-math.log(NN_dict[act[-1]][label])
            if(loss_sc=="sq"):
                loss1 = loss1+0.5*np.sum(((np.eye(1,output_size,label).T[:,0] -NN_dict[act[-1]])**2))
        loss1=loss1/images2.shape[0]
        #print(images1.shape[0])
        print(loss1)
        lossv.append(loss1)
        
        ypre1=[]
        for i in range(images1.shape[0]):
            ypre1.append(predict(images1[i]))
        from sklearn.metrics import accuracy_score
        errorT = 1-accuracy_score(labels1, ypre1)
        
        ypre2=[]
        for i in range(images11.shape[0]):
            ypre2.append(predict(images11[i]))
        from sklearn.metrics import accuracy_score
        errorV = 1-accuracy_score(labels11, ypre2)
        
        if j is 0:
            with open(expt_dir+"/log_train.txt", "w") as myfile:
                myfile.write("Epoch: %s, Step %s, Loss: %s, Error: %s, lr: %s \n" %(str(j), str(q), str(loss), str(errorT), str(alpha)))

            with open(expt_dir+"/log_val.txt", "w") as myfile:
                myfile.write("Epoch: %s, Step %s, Loss: %s, Error: %s, lr: %s\n" %(str(j), str(q), str(loss1), str(errorV), str(alpha)))
        else:
            with open(expt_dir+"/log_train.txt", "a") as myfile:
                myfile.write("Epoch: %s, Step %s, Loss: %s, Error: %s, lr: %s \n" %(str(j), str(q), str(loss), str(errorT), str(alpha)))

            with open(expt_dir+"/log_val.txt", "a") as myfile:
                myfile.write("Epoch: %s, Step %s, Loss: %s, Error: %s, lr: %s\n" %(str(j), str(q), str(loss1), str(errorV), str(alpha)))
            
        
        wt = []
        bt = []

        for u in range(len(weights)):
            wt.append(NN_dict[weights[u]])

        for u in range(len(bias)):
            wt.append(NN_dict[bias[u]])


        save_weights(np.array(wt),j+state+1)
        #save_bias(np.array(bt),j+state+1)
            
    if (anneal=="true"):
        if j>0:
            if errorV > prev_errorV:
                alpha = 0.5*alpha
        
    prev_errorV = errorV
    
    print("alpha",alpha)
    return losse , lossv


# NN_dict, bias, weights, pact,act = initialize(NN_dict,sizes)

# eps=0.9

# lossadam_t, lossadam_v , NN_dict = nag2(epocs,images1,NN_dict,batchsize,alpha,actFunc,labels1, bias, weights, pact,act, n,eps,images2, labels2)

# loss_t.append(lossadam_t)
# loss_v.append(lossadam_v)

# In[45]:


# epocs=6
# batchsize=100
# eta=0.9
# NN_dict=initialize(NN_dict)
# loss_nag1 , NN_dict= nag2(epocs,images[0:100] , NN_dict,batchsize,eta)


# ## Adam

# In[46]:


def adam(epocs,images1,NN_dict,batchsize,beta1,beta2,eps,alpha,actFunc,labels1, images11, labels11, bias, weights, pact,act, n):
    losse=[]
    lossv=[]
    prev_errorV = 0
    sumgradW=[]
    sumgradb=[]
    respropW=[]
    respropb=[]
    t=0
    loss = 0

    for t in range(0,images1.shape[0]):    
        input_x = images1[t]
        label = labels1[t] 
        NN_dict[pact[0]] = NN_dict[bias[0]]+np.matmul(NN_dict[weights[0]],input_x)
        #print("q",NN_dict[weights[0]].shape,input_x.shape)
        #print("a",NN_dict[pact[0]])
        NN_dict[act[0]] = activation(NN_dict[pact[0]],actFunc)
        #print("a2",NN_dict[act[0]])

        for k in range(1,n):
            #print("c",NN_dict[act[k-1]].shape)
            NN_dict[pact[k]] = NN_dict[bias[k]]+np.matmul(NN_dict[weights[k]],NN_dict[act[k-1]])
            #print("c",NN_dict[act[k-1]])
            NN_dict[act[k]] = activation(NN_dict[pact[k]],actFunc)
            #print(NN_dict[act[k]])
        NN_dict[pact[-1]] = NN_dict[bias[-1]]+np.matmul(NN_dict[weights[-1]],NN_dict[act[-2]])
        NN_dict[act[-1]] = output(NN_dict[pact[-1]])

        if(loss_sc=="ce"):
            loss = loss-math.log(NN_dict[act[-1]][label])
        if(loss_sc=="sq"):
            loss = loss+0.5*np.sum(((np.eye(1,output_size,label).T[:,0] -NN_dict[act[-1]])**2))
    loss=loss/images1.shape[0]
    #print(images1.shape[0])
    print(loss)
    losse.append(loss)


    loss1 = 0

    for t in range(0,images11.shape[0]):    
        input_x = images11[t]
        label = labels11[t] 
        NN_dict[pact[0]] = NN_dict[bias[0]]+np.matmul(NN_dict[weights[0]],input_x)
        #print("actFunc"+actFunc)
        NN_dict[act[0]] = activation(NN_dict[pact[0]],actFunc)

        for k in range(1,n):
            NN_dict[pact[k]] = NN_dict[bias[k]]+np.matmul(NN_dict[weights[k]],NN_dict[act[k-1]])
            NN_dict[act[k]] = activation(NN_dict[pact[k]],actFunc)

        NN_dict[pact[-1]] = NN_dict[bias[-1]]+np.matmul(NN_dict[weights[-1]],NN_dict[act[-2]])
        NN_dict[act[-1]] = output(NN_dict[pact[-1]])

        if(loss_sc=="ce"):
            loss1 = loss1-math.log(NN_dict[act[-1]][label])
        if(loss_sc=="sq"):
            loss1 = loss1+0.5*np.sum(((np.eye(1,output_size,label).T[:,0] -NN_dict[act[-1]])**2))
    loss1=loss1/images2.shape[0]
    #print(images1.shape[0])
    print(loss1)
    lossv.append(loss1)
    for j in range(0,epocs):
        gradWkVecSum = np.array([])
        gradbkVecSum = np.array([])
        for q in range(0,images1.shape[0],batchsize):
            for i in range(q,q+batchsize):
                input_x = images1[i]
                label = labels1[i]

                #Forward-Prop 

                NN_dict[pact[0]] = NN_dict[bias[0]]+np.matmul(NN_dict[weights[0]],input_x)
                NN_dict[act[0]] = activation(NN_dict[pact[0]],actFunc)

                for p in range(1,n):
                    #print("c",NN_dict[act[p-1]].shape)
                    NN_dict[pact[p]] = NN_dict[bias[p]]+np.matmul(NN_dict[weights[p]],NN_dict[act[p-1]])
                    #print("c",NN_dict[act[k-1]].shape)
                    NN_dict[act[p]] = activation(NN_dict[pact[p]],actFunc)
                    #dropout
                    random1 = np.random.binomial(1,0.9,NN_dict[act[p]].shape[0])
                    NN_dict[act[p]] = np.multiply(NN_dict[act[p]],random1)

                #print("a",p)
                NN_dict[pact[n]] = NN_dict[bias[n]]+np.matmul(NN_dict[weights[n]],NN_dict[act[n-1]])
                NN_dict[act[n]] = output(NN_dict[pact[n]])

                #print("a",NN_dict[pact[-1]])
                #Backward-Prop

                gradWkVec = []
                gradbkVec = []

                if(loss_sc=="ce"):
                    gradak = -(np.eye(1,output_size,labels1[i]).T[:,0] - NN_dict[act[-1]])
                if(loss_sc=="sq"):
                    gradak = -NN_dict[act[-1]][label]*(((np.eye(1,output_size,labels1[i]).T[:,0]- NN_dict[act[-1]]))**2)
                #print("x",NN_dict[act[-1]].shape)
                #print("w",gradak.shape)

                for k in range(n, 0, -1):
                    # Computing gradient w.r.t. parameters;
                    gradWk = np.matmul(gradak[np.newaxis].T,NN_dict[act[k-1]][np.newaxis])    
                    gradbk = gradak.copy()

                    #print(NN_dict[act[k-1]])

                    #print(act[k-1])

                    gradWkVec.append(gradWk)
                    gradbkVec.append(gradbk)

                    # Computing gradient w.r.t. layer below;
                    gradhkm1 = np.matmul(np.transpose(NN_dict[weights[k]]),gradak)

                    # Computing gradient w.r.t. layer below (pre-activation);
                    gradak = np.multiply(gradhkm1, gdash(NN_dict[pact[k-1]],actFunc))
                    #print("a",gradak.shape)
                    #print("b",gradhkm1.shape)
                    #print("v",gdash(NN_dict[pact[k-1]],actFunc).shape)

                gradWk = np.matmul(gradak[np.newaxis].T,input_x[np.newaxis])    
                gradbk = gradak.copy()

                #print(gradbk.shape)

                gradWkVec.append(gradWk)
                gradbkVec.append(gradbk)

                if i is 0:
                    gradWkVecSum = gradWkVec.copy()
                    gradbkVecSum = gradbkVec.copy()
                else:
                    for m in range(0,len(gradWkVecSum)):
                        gradWkVecSum[m] = gradWkVecSum[m] + gradWkVec[m]
                        gradbkVecSum[m] = gradbkVecSum[m] + gradbkVec[m]

            t=t+1
            #Update rule
            if(q is 0):
                if( j is 0):
                    sumgradW , sumgradb = gradWkVecSum.copy() , gradbkVecSum.copy()
            else:
                for m in range(0,len(gradWkVecSum)):
                    sumgradW[m] = (1-beta1)*gradWkVecSum[m] + beta1*sumgradW[m]
                    sumgradb[m] = (1-beta1)*gradbkVecSum[m] + beta1*sumgradb[m]
            if(q is 0):
                if( j is 0):
                    respropW , respropb = (np.square(gradWkVecSum)).copy() , (np.square(gradbkVecSum)).copy()
            else:
                for m in range(0,len(gradWkVecSum)):
                    respropW[m]=beta2*respropW[m]+(1-beta2)*np.square(gradWkVecSum[m])
                    respropb[m]=beta2*respropb[m]+(1-beta2)*np.square(gradbkVecSum[m])
            
            respropW1 , respropb1 = respropW.copy() , respropb.copy()
            sumgradW1 , sumgradb1 = sumgradW.copy() , sumgradb.copy()
            for m in range(0,len(gradWkVecSum)):
                sumgradW1[m] = sumgradW[m]*(1/(1-beta1**t))
                sumgradb1[m] = sumgradb[m]*(1/(1-beta1**t))
            for m in range(0,len(gradWkVecSum)):
                respropW1[m]=respropW[m]*(1/(1-beta2**t))
                respropb1[m]=respropb[m]*(1/(1-beta2**t))
            for l in range(0,n):
                NN_dict[weights[l]] = NN_dict[weights[l]] - alpha*np.divide(sumgradW1[n-l],np.sqrt(respropW1[n-l]+eps))
                NN_dict[bias[l]] = NN_dict[bias[l]] - alpha*np.divide(sumgradb1[n-l],np.sqrt(respropb1[n-l]+eps))

            
        
        #Loss-calc

        loss = 0

        for t in range(0,images1.shape[0]):    
            input_x = images1[t]
            label = labels1[t] 
            NN_dict[pact[0]] = NN_dict[bias[0]]+np.matmul(NN_dict[weights[0]],input_x)
            #print("q",NN_dict[weights[0]].shape,input_x.shape)
            #print("a",NN_dict[pact[0]])
            NN_dict[act[0]] = activation(NN_dict[pact[0]],actFunc)
            #print("a2",NN_dict[act[0]])

            for k in range(1,n):
                #print("c",NN_dict[act[k-1]].shape)
                NN_dict[pact[k]] = NN_dict[bias[k]]+np.matmul(NN_dict[weights[k]],NN_dict[act[k-1]])
                #print("c",NN_dict[act[k-1]])
                NN_dict[act[k]] = activation(NN_dict[pact[k]],actFunc)

            NN_dict[pact[-1]] = NN_dict[bias[-1]]+np.matmul(NN_dict[weights[-1]],NN_dict[act[-2]])
            NN_dict[act[-1]] = output(NN_dict[pact[-1]])

            if(loss_sc=="ce"):
                loss = loss-math.log(NN_dict[act[-1]][label])
            if(loss_sc=="sq"):
                loss = loss+0.5*np.sum(((np.eye(1,output_size,label).T[:,0] -NN_dict[act[-1]])**2))
        loss=loss/images1.shape[0]
        #print(images1.shape[0])
        print(loss)
        losse.append(loss)
        
        
        loss1 = 0

        for t in range(0,images11.shape[0]):    
            input_x = images11[t]
            label = labels11[t] 
            NN_dict[pact[0]] = NN_dict[bias[0]]+np.matmul(NN_dict[weights[0]],input_x)

            NN_dict[act[0]] = activation(NN_dict[pact[0]],actFunc)

            for k in range(1,n):
                NN_dict[pact[k]] = NN_dict[bias[k]]+np.matmul(NN_dict[weights[k]],NN_dict[act[k-1]])
                NN_dict[act[k]] = activation(NN_dict[pact[k]],actFunc)

            NN_dict[pact[-1]] = NN_dict[bias[-1]]+np.matmul(NN_dict[weights[-1]],NN_dict[act[-2]])
            NN_dict[act[-1]] = output(NN_dict[pact[-1]])

            if(loss_sc=="ce"):
                loss1 = loss1-math.log(NN_dict[act[-1]][label])
            if(loss_sc=="sq"):
                loss1 = loss1+0.5*np.sum(((np.eye(1,output_size,label).T[:,0] -NN_dict[act[-1]])**2))
        loss1=loss1/images2.shape[0]
        #print(images1.shape[0])
        print(loss1)
        lossv.append(loss1)
        
        ypre1=[]
        for i in range(images1.shape[0]):
            ypre1.append(predict(images1[i]))
        from sklearn.metrics import accuracy_score
        errorT = 1-accuracy_score(labels1, ypre1)
        
        ypre2=[]
        for i in range(images11.shape[0]):
            ypre2.append(predict(images11[i]))
        from sklearn.metrics import accuracy_score
        errorV = 1-accuracy_score(labels11, ypre2)
        
        if j is 0:
            with open(expt_dir+"/log_train.txt", "w") as myfile:
                myfile.write("Epoch: %s, Step %s, Loss: %s, Error: %s, lr: %s \n" %(str(j), str(q), str(loss), str(errorT), str(alpha)))

            with open(expt_dir+"/log_val.txt", "w") as myfile:
                myfile.write("Epoch: %s, Step %s, Loss: %s, Error: %s, lr: %s\n" %(str(j), str(q), str(loss1), str(errorV), str(alpha)))
        else:
            with open(expt_dir+"/log_train.txt", "a") as myfile:
                myfile.write("Epoch: %s, Step %s, Loss: %s, Error: %s, lr: %s \n" %(str(j), str(q), str(loss), str(errorT), str(alpha)))

            with open(expt_dir+"/log_val.txt", "a") as myfile:
                myfile.write("Epoch: %s, Step %s, Loss: %s, Error: %s, lr: %s\n" %(str(j), str(q), str(loss1), str(errorV), str(alpha)))
        
        wt = []
        bt = []

        for u in range(len(weights)):
            wt.append(NN_dict[weights[u]])

        for u in range(len(bias)):
            wt.append(NN_dict[bias[u]])

        print("s",len(wt))
        save_weights(np.array(wt),j+state+1)
        #save_bias(np.array(bt),j+state+1)
            
    if (anneal=="true"):
        if j>0:
            if errorV > prev_errorV:
                alpha = 0.5*alpha
        
    prev_errorV = errorV
    
    print("alpha",alpha)
    return losse , lossv



# In[47]:





# In[48]:




# #images2 = pca.transform(images2)

# ypre1=[]
# for i in range(images2.shape[0]):
#     ypre1.append(predict(images2[i]))
# from sklearn.metrics import accuracy_score
# print(accuracy_score(labels2, ypre1))


# In[64]:


# sizesList = np.array([[300,300,300]])
# sizesList = sizesList.astype(np.int64)


# In[70]:


loss_t =[]
loss_v =[]

ypre=[]
for i in range(images1.shape[0]):
    ypre.append(predict(images1[i]))
from sklearn.metrics import accuracy_score
print(accuracy_score(labels1, ypre))


# In[60]:




# In[122]:


# eta=0.01
# alpha=0.01
# epocs=100
# actfun='sigmoid'
# NN_dict=initialize(NN_dict)
# loss_nag=[]
# loss_nagepoc=[]
# finalepocs=500
# flag=1
# batch_size=20
# for i in range(final_epocs):
#     for j in range(sub_data.shape[0]):
#         if(j>batch_size*flag):
#             flag=flag+1
#             loss_moment1 , NN_dict= momentum(eta,epocs,images,alpha,NN_dict,actfun)
#             loss_moment.append(loss_moment1)
#     loss_momentepoch.append(loss_moment)


# In[52]:


# ypre=[]
# for i in range(images.shape[0]):
#     ypre.append(predict(images[i]))
# from sklearn.metrics import accuracy_score
# print(accuracy_score(labels, ypre))


# In[72]:


# data1 = pd.read_csv("valid.csv")

# sub_data1 = data1 #partial data
# sub_data1 = sub_data1.values
# print(sub_data1[0].shape)

# ids = sub_data1[:,1]
# print(ids.shape)

# images2 = sub_data1[:,1:-1]
# print(images2.shape)

# labels2 = sub_data1[:,-1]
# print(labels2.shape)

# labels2 = labels2.astype(np.int64)

# import sklearn
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(images2)
# images2 = scaler.transform(images2)

#images2 = pca.transform(images2)

ypre1=[]
for i in range(images2.shape[0]):
    ypre1.append(predict(images2[i]))
from sklearn.metrics import accuracy_score
print(accuracy_score(labels2, ypre1))

# ypre3=[]
# for i in range(images3.shape[0]):
#     ypre3.append(predict(images3[i]))

# ids3 = ids3.astype(np.int64)
# df = pd.DataFrame({'id':ids3,'label':ypre3})
# df.to_csv("submission99.csv", encoding="utf-8", index=False)


def main(opt,ids,images3,state):
    print(testing)
    
    if (testing == "false"):
    
        if (opt == "gd"):
    #         actFunc='sigmoid'

            #NN_dict, bias, weights, pact,act = initialize(NN_dict,sizes)

            print("gd")
            lossadam_t, lossadam_v = gradientdescent2(epocs,images1,NN_dict,batchsize,alpha,actFunc,labels1, bias, weights, pact,act, n,images2, labels2)

        if (opt == "momentum"):
            #actFunc='sigmoid'

    #         n = sizes.shape[0]

            #NN_dict, bias, weights, pact,act = initialize(NN_dict,sizes)

            eps=0.9

            lossadam_t, lossadam_v = momentum2(epocs,images1,NN_dict,batchsize,alpha,actFunc,labels1, bias, weights, pact,act, n,eps,images2, labels2)


        if (opt == "nag"):
            #actFunc = "sigmoid"
            #NN_dict, bias, weights, pact,act = initialize(NN_dict,sizes)

            eps=0.9

            lossadam_t, lossadam_v = nag2(epocs,images1,NN_dict,batchsize,alpha,actFunc,labels1, bias, weights, pact,act, n,eps,images2, labels2)

        if (opt == "adam"):
            eps=1e-8
            beta1=0.9
            beta2=0.999
            #actFunc='sigmoid'
            #NN_dict, bias, weights, pact,act = initialize(NN_dict,sizes)



            print("adam")
            lossadam_t, lossadam_v = adam(epocs,images1,NN_dict,batchsize,beta1,beta2,eps,alpha,actFunc,labels1, images2, labels2, bias, weights, pact,act, n)

        
        
    elif (testing == "true"):
        ypre3=[]
        
#         global id3
#         global images3
        
        for i in range(images3.shape[0]):
            ypre3.append(predict(images3[i]))

        #ids3 = ids3.astype(np.int64)
        df = pd.DataFrame({'id':ids,'label':ypre3})
        df.to_csv(expt_dir+"/prediction_"+"%s"%(str(state))+".csv", encoding="utf-8", index=False)
    return "Success"

            
#print(main(opt))


# ypre3=[]
# for i in range(images3.shape[0]):
#     ypre3.append(predict(images3[i]))

# ids3 = ids3.astype(np.int64)
# df = pd.DataFrame({'id':ids3,'label':ypre3})
# df.to_csv("submission52.csv", encoding="utf-8", index=False)

testFile = args.test

data3 = pd.read_csv(testFile)

print(data3.head(3))

sub_data3 = data3 #partial data
sub_data3 = sub_data3.values
print(sub_data3[0].shape)

ids3 = sub_data3[:,0]
#print("a",type(ids3))#.shape)

print(sub_data3[:,0])

images3 = sub_data3[:,1:]
print(images3.shape)

import sklearn
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(images3)
images3 = scaler.transform(images3)

images3=pca.transform(images3)

print(main(opt,sub_data3[:,0].astype(np.int64),images3,state))
