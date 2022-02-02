#Jake Zegeer
#G01056701

import random
import math
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

class kmeans:
    def __init__(self):
        self.iris_data = []
        self.image_clustering_data = []
    
    def readFiles(self):
        # read the iris data
        with open('iris_data/1616505271_4922109_iris_new_data.txt') as file:
            for line in file:
                data = []
                for num in line.split():
                    data.append(float(num))
                self.iris_data.append(data)
        # read the clustering data
        with open('image_clustering_data/1616506692_795991_new_test4.txt') as file:
            for line in file:
                data = []
                for num in line.split(','):
                    data.append(int(num))
                self.image_clustering_data.append(data)
    
    """This method returns the euclidean distance between two points"""
    def euclideanDistance(self,point1, point2):
        distance = 0
        for i in range(len(point1)):
            #Euclidean distance formula
            distance += ((point1[i]-point2[i])**2)
        euclidean_dist = math.sqrt(distance)
        return euclidean_dist

    """This method returns centroid for clusters list"""
    def clusterCentroid(self,items, prec):
        centroids_list = []
        for item in items:
            item_list = []
            for index in range(len(item[0])):
                mean_list = []
                for  itm in item:
                    mean_list.append(itm[index])
                item_list.append(round(np.mean(mean_list), prec))
            centroids_list.append(item_list)
        return centroids_list

    """This method returns false when the two lists are not same otherwise returns False"""
    def comparePointLists(self,list_1, list_2):
        for item in list_1:
            if item not in list_2:
                return False
        return True

    """Fit the kmeans"""
    def kmeansFit(self,data, k=3,max_iter=500, prec=5):
        random_index = random.sample(range(len(data)), k)   # Randomly selects initial numbers for centroids
        centroids = [data[index] for index in random_index] # Initialization of initial centroids using random_index
        clusters = [[] for i in range(k)]
        count = 0
        while count < max_iter:
            count += 1
            # Initialization of clusters of empty list
            for point in data:
                # Calculate the euclidean distance between each point in the data set and the centroid
                euclidean_distances = [self.euclideanDistance(point, centroid) for centroid in centroids]
                clusters[euclidean_distances.index(min(euclidean_distances))].append(point)
            new_centroids = self.clusterCentroid(clusters, prec) # Calculate the new centroid
            if self.comparePointLists(centroids, new_centroids):
                break
            centroids = new_centroids # Updating new centroids
            #print(count)
        return clusters

    """Predict the cluster values"""
    def yPredictKmeans(self,data,k):
        clusters = self.kmeansFit(data,k)
        y_predict = []
        for value in data:
            for i in range(len(clusters)):
                if value in clusters[i]:
                    y_predict.append(i+1)
                    break
        return y_predict

    """This method is scaling the image range -1 to 1"""
    def preProcessing(self,image):
        new_image = []
        for pixel in image:
            value = round(pixel/255 - 1,5)
            new_image.append(value)
        return new_image

    def writeFile(self,filename,y_pred):
        # write the image clustering format
        with open(filename, 'w') as file:
            for list_item in y_pred:
                file.write('%s\n' % str(list_item))
    
    #iris clustering without feature reduction
    def irisData(self):
        #Predict value using 3 clusters
        y_pred = self.yPredictKmeans(self.iris_data,3)
        self.writeFile('iris_label_format.txt',y_pred)
        
    '''
    #Iris data with feature reduction 
    def imageIrisData(self):
        iris_data = [self.preProcessing(image) for image in self.iris_data]
        #It's redeuce the dimension into 2
        X_iris_data = TSNE(n_components=2).fit_transform(iris_data)
        iris_data = X_iris_data.tolist()
        y_pred = self.yPredictKmeans(iris_data,3)
        self.writeFile('image_clustering_label_format.txt',y_pred)
        #Plot the graph then save the figure
        plt.figure(figsize=(3, 3))
        plt.scatter(X_iris_data[:, 0], X_iris_data[:, 1],c=y_pred)
        plt.title("Clusters using kmeans")
        plt.xlabel("TSNE_1")
        plt.ylabel("TSNE_2")
        plt.savefig('iris_plot.png')
        '''
        

#Dimension reduction
    def imageClusteringData(self):
        # Pre processing
        image_clustering_data = [self.preProcessing(image) for image in self.image_clustering_data]
        #It's reduce the dimension into 2
        X_image_clustering_data = TSNE(n_components=2).fit_transform(image_clustering_data)
        image_clustering_data = X_image_clustering_data.tolist()
        #Predict value using 10 clusters
        y_pred = self.yPredictKmeans(image_clustering_data,14)
        self.writeFile('image_clustering_label_format.txt',y_pred)
        #Plot the graph then save figure
        plt.figure(figsize=(5, 5)) 
        plt.scatter(X_image_clustering_data[:, 0], X_image_clustering_data[:, 1],c=y_pred)
        plt.title("Clusters using kmeans")
        plt.xlabel("TSNE_1")
        plt.ylabel("TSNE_2")
        plt.savefig('image_clustering_Plot.png')


if __name__ == '__main__':
    obj = kmeans()
    obj.readFiles()
    obj.irisData()
    #obj.imageIrisData()
    obj.imageClusteringData()
    

