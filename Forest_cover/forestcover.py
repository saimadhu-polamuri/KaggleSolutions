# Solution for kaggle forestcover problem 
__autor__ = 'saimadhu Polamuri'
__website__ = 'www.dataaspirant.com'
__createdon__ = '16-Feb-2015'

import pandas as pd
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

class ForestCover(object):

	""" ForestCover class is to predict ForestCover type from a range of 1 to 7 """

	def __init__(self,train_data,test_data):

		""" Initial funtion to load train and test data """

		self.train_data = train_data
		self.test_data = test_data
		#print self.train_data
	def convert_xy_traindata(self):

		""" Converts train data to x_parameters and y_parameters data """

		x_train = self.train_data[['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points','Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4','Soil_Type1','Soil_Type2','Soil_Type3','Soil_Type4','Soil_Type5','Soil_Type6','Soil_Type7','Soil_Type8','Soil_Type9','Soil_Type10','Soil_Type11','Soil_Type12','Soil_Type13','Soil_Type14','Soil_Type15','Soil_Type16','Soil_Type17','Soil_Type18','Soil_Type19','Soil_Type20','Soil_Type21','Soil_Type22','Soil_Type23','Soil_Type24','Soil_Type25','Soil_Type26','Soil_Type27','Soil_Type28','Soil_Type29','Soil_Type30','Soil_Type31','Soil_Type32','Soil_Type33','Soil_Type34','Soil_Type35','Soil_Type36','Soil_Type37','Soil_Type38','Soil_Type39','Soil_Type40']]
		y_train = self.train_data['Cover_Type']

		return x_train,y_train

	def randomforest_classifier_model(self,x_parameters,y_parameters):

		""" Fit RandomForestClassifier model with tarin data """

		self.x_parameters = x_parameters
		self.y_parameters = y_parameters

		x_train = self.train_data[['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points','Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4','Soil_Type1','Soil_Type2','Soil_Type3','Soil_Type4','Soil_Type5','Soil_Type6','Soil_Type7','Soil_Type8','Soil_Type9','Soil_Type10','Soil_Type11','Soil_Type12','Soil_Type13','Soil_Type14','Soil_Type15','Soil_Type16','Soil_Type17','Soil_Type18','Soil_Type19','Soil_Type20','Soil_Type21','Soil_Type22','Soil_Type23','Soil_Type24','Soil_Type25','Soil_Type26','Soil_Type27','Soil_Type28','Soil_Type29','Soil_Type30','Soil_Type31','Soil_Type32','Soil_Type33','Soil_Type34','Soil_Type35','Soil_Type36','Soil_Type37','Soil_Type38','Soil_Type39','Soil_Type40']]
		y_train = self.train_data['Cover_Type']
		clf = RandomForestClassifier(n_estimators=10)
		clf = clf.fit(self.x_parameters, self.y_parameters)
		
		return clf 

	def predict_test_data(self,model_type):

		""" predicts test data using train data randomforest classifier model """

		self.model_type = model_type

		#x_test = self.test_data[['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points','Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4','Soil_Type1','Soil_Type2','Soil_Type3','Soil_Type4','Soil_Type5','Soil_Type6','Soil_Type7','Soil_Type8','Soil_Type9','Soil_Type10','Soil_Type11','Soil_Type12','Soil_Type13','Soil_Type14','Soil_Type15','Soil_Type16','Soil_Type17','Soil_Type18','Soil_Type19','Soil_Type20','Soil_Type21','Soil_Type22','Soil_Type23','Soil_Type24','Soil_Type25','Soil_Type26','Soil_Type27','Soil_Type28','Soil_Type29','Soil_Type30','Soil_Type31','Soil_Type32','Soil_Type33','Soil_Type34','Soil_Type35','Soil_Type36','Soil_Type37','Soil_Type38','Soil_Type39','Soil_Type40']]
		x_train = self.train_data[['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points','Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4','Soil_Type1','Soil_Type2','Soil_Type3','Soil_Type4','Soil_Type5','Soil_Type6','Soil_Type7','Soil_Type8','Soil_Type9','Soil_Type10','Soil_Type11','Soil_Type12','Soil_Type13','Soil_Type14','Soil_Type15','Soil_Type16','Soil_Type17','Soil_Type18','Soil_Type19','Soil_Type20','Soil_Type21','Soil_Type22','Soil_Type23','Soil_Type24','Soil_Type25','Soil_Type26','Soil_Type27','Soil_Type28','Soil_Type29','Soil_Type30','Soil_Type31','Soil_Type32','Soil_Type33','Soil_Type34','Soil_Type35','Soil_Type36','Soil_Type37','Soil_Type38','Soil_Type39','Soil_Type40']]
		model = self.model_type
		#predicted_results = model.predict(x_test)
		predicted_results = model.predict(x_train)
		
		return predicted_results

	def check_predicted_model(self,predicted_results):

		""" Checks the efficiency of prdicted_model """

		self.predicted_results = predicted_results

		y_train = self.train_data['Cover_Type']
		count = 0
		for i in xrange(0,len(self.predicted_results)):
			if y_train[i] == self.predicted_results[i]:
				count +=1
		
		return count

	def svm_linearkernel(self,x_parameters,y_parameters):

		""" Support vector mechine with linear kernel to fit our train data """

		self.x_parameters = x_parameters
		self.y_parameters = y_parameters

		C = 1.0  # SVM regularization parameter
		svc = svm.SVC(kernel='linear', C=C).fit(self.x_parameters, self.y_parameters)
		
		return svc

	def svm_rbfkernel(self,x_parameters,y_parameters):

		""" Support vector mechine with Radial basis function kernel to fit our train data """

		slef.x_parameters = x_parameters
		self.y_parameters = y_parameters

		C = 1.0  # SVM regularization parameter
		rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(self.x_parameters, self.y_parameters)
		
		return rbf_svc

	def svm_polykernel(self,x_parameters,y_parameters):

		""" Support vector mechine with poly kernel to fit our train data """

		self.x_parameters = x_parameters
		self.y_parameters = y_parameters

		C = 1.0  # SVM regularization parameter
		poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(self.x_parameters, self.y_parameters)

		return poly_svc

	def svm_linear(self,x_parameters,y_parameters):

		""" Support vector mechine wiht linear svm model to fit our train data """

		self.x_parameters = x_parameters
		self.y_parameters = y_parameters

		C = 1.0  # SVM regularization parameter
		lin_svc = svm.LinearSVC(C=C).fit(X, y)

	def save_predicted_results(self,predicted_results):

		""" saves predicted resutls for test data """

		self.predicted_results = predicted_results

		id_value = self.test_data['Id']
		f = open("prediction.csv","w")
		f.write("Id,Cover_Type\n")
		for i in xrange(len(self.predicted_results)):
			f.write("{},{}\n".format(id_value[i], self.predicted_results[i]))
		f.close()

def main():

	train_csv = pd.read_csv('train.csv')
	test_csv = pd.read_csv('test.csv')
	#print len(test_csv)
	print len(train_csv)
	forestcover = ForestCover(train_csv,test_csv)
	x_train,y_train = forestcover.convert_xy_traindata()
	# svm_linearkernel_model = forestcover.svm_linearkernel(x_train,y_train)
	# svm_rbfkernel_model = forestcover.svm_rbfkernel(x_train,y_train)
	# svm_polykernel_model = forestcover.svm_polykernel(x_train,y_train)
	# svm_linear_model = forestcover.svm_linear(x_train,y_train)
	randomforest_model = forestcover.randomforest_classifier_model(x_train,y_train)
	results = forestcover.predict_test_data(randomforest_model)
	print len(results)
	#print forestcover.check_predicted_model(results)
	#forestcover.save_predicted_results(results)
	#print forestcover.train_data
if __name__ == "__main__":
	main()