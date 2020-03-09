import numpy as np
import math
import operator

class KNN:
    def __init__(self, k):
        # KNN state here
        # Feel free to add methods
        self.k = k

    def distance(self, featureA, featureB):
        diffs = (featureA - featureB) ** 2
        return np.sqrt(diffs.sum())

    def train(self, X, y):
        # training logic here
        # input is an array of features and labels
        self.X_train = X
        self.y_train = y

    def k_neighbors_list(self, Xtest):
        distance_list = []
        i = 0
        neighbors = []
        for t in self.X_train:
            d = self.distance(t, Xtest) #calculate distance between target and neighbour
            distance_list.append((d, self.y_train[i])) #store the values
            i += 1

        def key1(x):
            return x[0]

        list = sorted(distance_list, key=key1) #sort in descending order

        for i in range(self.k): #store the k nearest neighbors
            neighbors.append(list[i][1])
        return neighbors

    def predict(self, X):
        zero = []
        for l in range(len(X)):
            k_neighbors = self.k_neighbors_list(X[l]) #find k neighbors for each sample
            o = 0
            z = 0
            for l in range(len(k_neighbors)):
                if (k_neighbors[l] == 0): #classify based on neighbors
                    z += 1
                else:
                    o += 1
            if z > o: #check to which class majority of k neighbors belong to
                zero.append(0)
            else:
                zero.append(1)
        return np.array(zero)


class ID3:
    tree={}
    def __init__(self, nbins, data_range):
        # Decision tree state here
        # Feel free to add methods
        self.bin_size = nbins
        self.range = data_range

    def preprocess(self, data):
        # Our dataset only has continuous data
        norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
        categorical_data = np.floor(self.bin_size * norm_data).astype(int)
        return categorical_data

    def train(self, X, y):
        # training logic here
        # input is array of features and labels
        categorical_data = self.preprocess(X)
        attributes = []
        for i in range(len(X[0])): #create a set of attribute names for usage
            attributes.append(i)

        y_train = y.tolist()
        self.tree=self.tree_building(categorical_data, attributes, y_train) #build the tree

    def tree_building(self, data, attributes, y_train):
        unique = {}

        for t in data: #to find the class to which majority of samples belong to
            if (t[-1]) in unique:
                unique[t[-1]] += 1
            else:
                unique[t[-1]] = 1

        majority_class = max(unique.items(), key=operator.itemgetter(1))[0]
        default_class_value=majority_class #put it as default class

        if ((len(attributes) - 1)) <= 0: #if no attributes return default value
            return default_class_value
        class_value=y_train[0]
        f=0
        for each in y_train:
            if(each==class_value):
                f+=1
            else:
                break
        if f==len(y_train):#if all the samples belong to same class return the same class value
            return y_train[0]
        elif len(data)<=0:  #no samples left
            return default_class_value
        else:
            best = self.best_attribute(data, attributes) #find best_split_attribute
            tree = {best: {}}
            index = attributes.index(best)
            #get the unique sample values under the best attribute category
            sample_values = []
            for each in data:
                if each[index] in sample_values:
                    continue
                else:
                    sample_values.append(each[index])
            #get sample values after splitting with best attribute
            for s in sample_values:
                new_d = []
                for each in data:
                    if (each[index] != s):
                        continue
                    else:
                        new = []
                        for i in range(0, len(each)):
                            if (i != index):
                                new.append(each[i])
                        new_d.append(new)
                new_attr_list=[]
                for k in attributes:
                    new_attr_list.append(k)
                new_attr_list.remove(best) #remove the best attribute from further classification
                subtree = self.tree_building(new_d, new_attr_list, y_train) #build tree with new attribute and subset
                tree[best][s] = subtree

        return tree

    def best_attribute(self, data, attributes):

        best = attributes[0]
        gain={}
        names=[]

        for i in range(len(attributes)): #find gain for each attribute and store
            newGain = self.info_gain(attributes, data, attributes[i])
            gain[attributes[i]]=newGain
            names.append(attributes[i])
        max1=max(gain.items(), key=operator.itemgetter(1))[0] #find max gain value

        for key, value in gain.items(): #return the attribute corresponding to highest info gain value
            if value == max1:
                best=key
                break

        return best

    def info_gain(self, attributes, data, attr):

        counter = {}
        i = attributes.index(attr)

        for entry in data: #find the unique categories within the attribute selected
            if (entry[i]) in counter:
                counter[entry[i]] += 1.0
            else:
                counter[entry[i]] = 1.0

        gained = 0

        for val in counter.keys(): #calculate info gain for this attribute
            a1=counter[val]
            a2=sum(counter.values())
            probability = a1/a2
            subset = self.get_subset(data,val,i)
            gained = gained + probability * self.entropy(subset)

        total=self.entropy(data) #find total net gain of info due to split with this attribute and return
        info_gain= total - gained

        return info_gain

    def get_subset(self,data,value,i):
        record= [e for e in data if e[i] == value] #find the subset data after split
        return record

    def entropy(self, data):

        counter = {}
        for entry in data:
            if (entry[-1]) in counter:
                counter[entry[-1]] += 1.0
            else:
                counter[entry[-1]] = 1.0

        entropy_val = 0

        for c in counter.values():
            a1= (-c / len(data))
            a2= math.log(c / len(data), 2)
            a3=a1*a2
            entropy_val= entropy_val+ a3

        return entropy_val

    def predict(self, X):
        # Run model here
        # Return array of predictions where there is one prediction for each set of features
        categorical_data = self.preprocess(X)
        attributes = []
        end = []
        l = 0

        for i in range(30):
            attributes.append(i)

        for each in categorical_data:
            result=""
            temp = self.tree
            l = l + 1
            #traverse the tree
            k=0
            while (k==0):
                locate=list(temp.keys())[0]
                root = tree_node(locate, temp[locate])
                temp = temp[locate]
                i = attributes.index(root.value)
                value = each[i]
                if (value in temp.keys()):
                    result = temp[value]
                    temp = temp[value]
                else:
                    result = "leaf"
                    break
                if isinstance(temp,dict):
                    k=0
                else:
                    k=1

            end=self.make_prediction(result,each,end)

        return np.array(end)


    def make_prediction(self,result,each,end):
        if result != "leaf":
            if result == each[-1]:

                end.append(1)
            else:

                end.append(0)
        else:
            end.append(0)
        return end

class tree_node():
    subtrees = []

    def __init__(self, val, tree):
        self.value = val
        if (isinstance(tree, dict)):
            child_keys=tree.keys()
            self.subtrees = child_keys
    def is_leaf(self):
        return 0



class Perceptron:
    def __init__(self, w, b, lr):
        # Perceptron state here, input initial weight matrix
        # Feel free to add methods
        self.lr = lr
        self.w = w
        self.b = b

    def sigmoid(self, x):
        xx = float(x)
        c = 1 / (1 + np.exp(-xx))
        p = self.calc(c)
        return p

    def calc(self, x):
        if x > 0:
            return 1
        else:
            return 0

    def train(self, X, y, steps):
        # training logic here
        # input is array of features and labels
        t = 0
        i = 0
        while t < steps:
            if (i == len(y)):
                i = 0
            xi, di = X[i], y[i]
            yi = self.sigmoid(np.dot(xi, self.w) + self.b)
            if yi != di:
                self.w = self.w + (self.lr * (di - yi) * xi)
            t = t + 1
            i = i + 1

    def predict(self, X):
        # Run model here
        # Return array of predictions where there is one prediction for each set of features

        zero = []
        for i in range(len(X)):
            c = self.sigmoid(np.dot(X[i], self.w) + self.b)
            if c > 0:
                zero.append(1)
            else:
                zero.append(0)
        return np.array(zero)


class MLP:
    def __init__(self, w1, b1, w2, b2, lr):
        self.l1 = FCLayer(w1, b1, lr)
        self.a1 = Sigmoid()
        self.l2 = FCLayer(w2, b2, lr)
        self.a2 = Sigmoid()

    def MSE(self, prediction, target):
        return np.square(target - prediction).sum()

    def MSEGrad(self, prediction, target):
        return - 2.0 * (target - prediction)

    def shuffle(self, X, y):
        idxs = np.arange(y.size)
        np.random.shuffle(idxs)
        return X[idxs], y[idxs]

    def train(self, X, y, steps):

        for s in range(steps):
            i = s % y.size
            if (i == 0):
                X, y = self.shuffle(X, y)

            xi = np.expand_dims(X[i], axis=0)

            yi = np.expand_dims(y[i], axis=0)
            pred = self.l1.forward(xi)
            pred = self.a1.forward(pred)
            pred = self.l2.forward(pred)
            pred = self.a2.forward(pred)
            loss = self.MSE(pred, yi)

            grad = self.MSEGrad(pred, yi)
            grad = self.a2.backward(grad)
            grad = self.l2.backward(grad)
            grad = self.a1.backward(grad)
            grad = self.l1.backward(grad)

    def predict(self, X):
        pred = self.l1.forward(X)
        pred = self.a1.forward(pred)
        pred = self.l2.forward(pred)
        pred = self.a2.forward(pred)
        pred = np.round(pred)
        return np.ravel(pred)


class FCLayer:
    stack1 = []

    def __init__(self, w, b, lr):
        self.lr = lr
        self.w = w  # Each column represents all the weights going into an output node
        self.b = b

    def forward(self, input):
        # Write forward pass here
        self.stack1.append(input)
        c = (np.dot(input, self.w) + self.b)
        return c

    def backward(self, gradients):
        # Write backward pass here

        x = self.stack1.pop()

        xt = x.transpose()

        w_ = np.dot(xt, gradients)
        wt = self.w.transpose()
        x_ = np.dot(gradients, wt)
        self.w = self.w - self.lr * w_
        self.b = self.b - self.lr * gradients

        return x_


class Sigmoid:
    stack2 = []

    def __init__(self):
        None

    def sigmoid(self, i):
        g = 1 / (1 + np.exp(i))
        return g

    def forward(self, input):
        self.stack2.append(input)
        # Write forward pass here
        c = self.sigmoid(input)
        return c

    def backward(self, gradients):
        # Write backward pass here
        p = self.stack2.pop()
        c = (gradients) * (1 - self.sigmoid(p)) * (self.sigmoid(p))
        return c
