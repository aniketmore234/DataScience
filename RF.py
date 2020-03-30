def K_u_v( U, V, kernel, kernel_param = 1 ):
  if ( kernel == 'linear' ):
      return( np.matmul( U, V.T ) )
  elif( kernel == 'poly' ):
      return( ( 1 + np.matmul( U, V.T ) )**int( kernel_param ) )

class RandomForest():
    def fit(self, x, y, n_trees, n_features, sample_sz, depth=5, min_leaf=5, kernel='poly',kernel_param=1):
        np.random.seed(12)
        if n_features == 'sqrt':
            self.n_features = int(np.sqrt(x.shape[1]))
        elif n_features == 'log2':
            self.n_features = int(np.log2(x.shape[1]))
        else:
            self.n_features = n_features
        #print(self.n_features, "sha: ",x.shape[1])   
        
        self.kernel = kernel
        self.kernel_param = kernel_param
        self.reg = LinearRegr()
        
        self.x, self.y, self.sample_sz, self.depth, self.min_leaf  = x, y, sample_sz, depth, min_leaf
        self.trees = [self.create_tree() for i in range(n_trees)]
        return self

    def create_tree(self):
        idxs = np.random.permutation(len(self.y))[:int(self.x.shape[0]*self.sample_sz)]
        f_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        return DecisionTree(self.x[idxs], self.y[idxs], self.n_features, f_idxs, idxs = np.array(range(int(self.x.shape[0]*self.sample_sz))), reg=self.reg, depth = self.depth, min_leaf=self.min_leaf, kernel=self.kernel,kernel_param=self.kernel_param)
        
    def predict(self, x):
        return np.mean([t.predict(x) for t in self.trees], axis=0)

def find_nearest(array, value):
    array = np.asarray(array)
    id = (np.abs(array - value)).argmin()
    return array[id]
  
class DecisionTree():
    def __init__(self, x, y, n_features, f_idxs, idxs, reg, depth=5, min_leaf=5,  kernel='poly', kernel_param=1):
        self.x, self.y, self.idxs, self.min_leaf, self.f_idxs = x, y, idxs, min_leaf, f_idxs
        self.depth = depth
        self.shape = x[idxs].shape[0]
        
        self.n_features = n_features
        self.n, self.c = len(idxs), x.shape[1]

        self.kernel = kernel
        self.kernel_param = kernel_param
        
        self.reg = reg
        self.reg.fit(x[idxs],y[idxs],kernel='poly', kernel_param=self.kernel_param)
        
        self.x_idxs, self.y_idxs = x[idxs], y[idxs]
        
        self.val = self.reg.coef_ #,self.reg.intercept_]
        
        #print(self.val.shape)
        self.score = float('inf')
        self.find_varsplit()

        
        
    def find_varsplit(self):
        for i in self.f_idxs: self.find_better_split(i)
        if self.is_leaf: return
        x = self.split_col
        lhs = np.nonzero(x<=self.split)[0]
        rhs = np.nonzero(x>self.split)[0]
        
        #print(lhs.shape[0],rhs.shape[0],self.var_idx,self.split)
        
        lf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        rf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        
        self.lhs = DecisionTree(self.x, self.y, self.n_features, lf_idxs, self.idxs[lhs], self.reg, depth=self.depth-1, min_leaf=self.min_leaf, kernel =self.kernel, kernel_param = self.kernel_param)
        
        self.rhs = DecisionTree(self.x, self.y, self.n_features, rf_idxs, self.idxs[rhs], self.reg, depth=self.depth-1, min_leaf=self.min_leaf, kernel =self.kernel, kernel_param = self.kernel_param)
        
        
    def find_better_split(self, var_idx):
        x_var, y_var = self.x[self.idxs,var_idx], self.y[self.idxs] #np.sort(self.x[self.idxs,var_idx]), np.sort(self.y[self.idxs])
        x_lhs, y_lhs, x_rhs, y_rhs = self.x_idxs[x_var<=np.median(x_var)], self.y_idxs[x_var<=np.median(x_var)], self.x_idxs[x_var>np.median(x_var)], self.y_idxs[x_var>np.median(x_var)]
        lhs_cnt, rhs_cnt = x_lhs.shape[0], x_rhs.shape[0]
        
        #if x_var.shape[0]<=self.min_leaf:
        #  return
        
        if lhs_cnt>0 and rhs_cnt>0:
          regL = self.reg #LinearRegression()
          regL.fit(x_lhs,y_lhs, self.kernel, self.kernel_param)
          lhs_mse = mean_squared_error(regL.predict(x_lhs),y_lhs)
          regR = self.reg #LinearRegression()
          regR.fit(x_rhs,y_rhs, self.kernel, self.kernel_param)
          rhs_mse = mean_squared_error(regR.predict(x_rhs),y_rhs)
        else:
          return
        
        curr_score = (lhs_cnt*lhs_mse+rhs_cnt*lhs_mse)/(lhs_cnt+rhs_cnt)
                
        if curr_score<self.score: 
            self.var_idx, self.score, self.split = var_idx, curr_score, find_nearest(x_var, np.median(x_var))
    
    @property
    def split_name(self): return self.x[self.var_idx]
    
    @property
    def split_col(self): return self.x[self.idxs,self.var_idx]

    @property
    def is_leaf(self):  return self.score == float('inf') or self.depth <= 0 or self.shape <= self.min_leaf 
    

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def phiFunc(self,x):
        poly = PolynomialFeatures(self.kernel_param)
        #print(poly.fit_transform(x).shape)
        return poly.fit_transform(x)

    def predict_row(self, xi):
        #print(xi.reshape(-1, 1).shape)
        if self.is_leaf: return  np.matmul(self.phiFunc(xi.reshape(-1, 1).T),self.val) #np.dot(self.val[0],xi)+self.val[1]
        t = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return t.predict_row(xi)
