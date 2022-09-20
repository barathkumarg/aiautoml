from flask import Flask,flash, render_template, request, redirect, url_for,send_file, session,Response,flash,jsonify,json
import os
import math
import numpy as np
import statistics
import pandas as pd
from werkzeug.utils import secure_filename


#ml dependencies
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import KFold
kf = KFold(n_splits=10, random_state=None)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import joblib
global df,output
app = Flask(__name__,static_url_path='',
            static_folder='../static',
            template_folder='../templates')
app.secret_key = 'your secret key'

from app import login,logout,charts,eda,model_building,predictions,view_table,download_table,import_pickle,test_pickle


