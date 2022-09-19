from flask import Flask,flash, render_template, request, redirect, url_for, session,Response,flash,jsonify,json
import os
import math
import numpy as np
import statistics
import pandas as pd
from werkzeug.utils import secure_filename
from ml_requirements import *
global df

app = Flask(__name__)
app.secret_key = 'your secret key'
@app.route('/',methods=['POST','GET'])
def login():
    if request.method == 'POST':
        project_name=request.form['pname']
        type=request.form['type']
        file=request.files['file']
        file = request.files['file']
        print(file)
        filename = secure_filename(file.filename)
        file.save(secure_filename(file.filename))
        session['name']=project_name
        session['type']=type
        session['file']=filename
        session['demo_trigger'] = False  #To view the demo model page
        return redirect(url_for('home'))


    else:
        return render_template('login.html')


@app.route('/home',methods=['POST','GET'])
def home():
    if 'file' in session:

        filename=session.get('file')
        project_name=session.get('name')
        type=session.get('type')
        df=pd.read_csv(filename)#get the csv data
        cols=df.columns#csv columns
        cols_type=list(df.dtypes)
        #print(type(cols_type))
        #print(cols_type)
        row_len=[]
        nan_len=[]
        for i in cols:#count the null and correct values
            lenght=df[i].count()
            nan_lenght=df[i].isnull().sum()
            nan_len.append(nan_lenght)
            row_len.append(lenght)

        nan_=(sum(nan_len)/sum(row_len))*100 #get the correct and null values
        nan_=round(nan_,2)

        len1=row_len[0]+nan_len[0]

        #central tendency
        #mean-median-mode
        ct=[]
        c=[]
        for i in cols:
            if df[i].dtype=='object':
                c.append('Str')
            else:
                c.append(round(df[i].mean(),2))
        ct.append(c)
        #median
        c=[]
        for i in cols:
            if df[i].dtype == 'object':
                c.append('Str')
            else:
                c.append(round(df[i].median(),2))
        ct.append(c)

        #mode
        c = []
        for i in cols:
            if df[i].dtype == 'object':
                c.append('Str')
            else:
                c.append(statistics.mode(df[i]))
        ct.append(c)


        #calculate outlier percentage
        outlier=[]
        count=0
        for i in cols:
            if (df[i].dtype!='object'):
                percentile25 = df[i].quantile(0.25)
                percentile75 = df[i].quantile(0.75)

                iqr = percentile75 - percentile25
                upper_limit = percentile75 + 1.5 * iqr
                sum_=(df[i]>upper_limit).sum()
                if sum_>0:
                    outlier.append(i)
                count=count+sum_
        outlier_per=round((count/(len1*len(cols)))*100,2)

        trigger = session.get('demo_trigger')
        return render_template('index.html',cols=cols,cols_type=cols_type,row_len=row_len,nan_len=nan_len,len=len(cols),nan_=nan_,project_name=project_name,type=type,ct=ct,outlier=outlier,outlier_per=outlier_per,trigger=trigger)
    else:
        return render_template('login.html')


@app.route('/replace_nan',methods=['POST','GET'])
def replace_nan():
    if 'file' in session:
        type=request.args.get('type')
        cols=request.args.get('cols')
        # data change Code

        filename = session.get('file')
        df = pd.read_csv(filename)

        if type=='mean':
            df[cols].fillna(value=round(df[cols].mean(),2),inplace=True)

        elif type=='median':
            df[cols].fillna(value=round(df[cols].median(),2),inplace=True)
        elif type=='mode':
            df[cols].fillna(value=statistics.mode(df[cols]), inplace=True)

        df.to_csv('{0}'.format(filename),index=False)
        flash("you are successfuly logged in")
        return redirect(url_for('home'))
    else:
        return render_template('login.html')
#table page
@app.route('/table_page',methods=['POST','GET'])
def table_page():
    if 'file' in session:

        filename=session.get('file')
        project_name=session.get('name')
        type=session.get('type')
        df=pd.read_csv(filename)#get the csv data
        cols=df.columns#csv columns
        row_len = []
        nan_len = []
        for i in cols:#count the null and correct values
            lenght=df[i].count()
            nan_lenght=df[i].isnull().sum()
            nan_len.append(nan_lenght)
            row_len.append(lenght)
        len1 = row_len[0] + nan_len[0]

        trigger = session.get('demo_trigger')
        return render_template('table.html', cols=cols, data=df,project_name=project_name,type=type,len1=len1,trigger=trigger)

#drop_dummies function
@app.route('/drop_dummies',methods=['POST','GET'])
def drop_dummies():
    if 'file' in session:
        type=request.args.get('type')
        cols=request.args.get('cols')
        filename = session.get('file')
        df = pd.read_csv(filename)
        if type=='drop':

            df.drop([cols], axis=1,inplace=True)
            df.to_csv('{0}'.format(filename), index=False)
            flash("you are successfuly logged in")

            return redirect(url_for('home'))
        elif type=='dummies':
            label_encoder = preprocessing.LabelEncoder()
            df[cols] = label_encoder.fit_transform(df[cols])
            df.to_csv('{0}'.format(filename), index=False)
            return redirect(url_for('home'))
        elif type=='dummies1':
            df = df[df[cols].notna()]
            df.to_csv('{0}'.format(filename), index=False)
            return redirect(url_for('home'))
    else:
        return render_template('login.html')

#remove outlier
@app.route('/remove_outlier',methods=['POST','GET'])
def remove_outlier():
    if 'file' in session:

        cols=request.args.get('cols')
        filename = session.get('file')
        df = pd.read_csv(filename)
        percentile25 = df[cols].quantile(0.25)
        percentile75 = df[cols].quantile(0.75)

        iqr=percentile75-percentile25
        upper_limit = percentile75 + 1.5 * iqr


        df.drop(df.loc[df[cols]>upper_limit].index, inplace=True)
        df.to_csv('{0}'.format(filename), index=False)
        return redirect(url_for('home'))
    else:
        return render_template('login.html')




#scatter_chart page
@app.route('/chart',methods=['POST','GET'])
def chart():
    if 'file' in session and request.method == 'POST':
        x_axis=request.form['x']
        y_axis=request.form['y']
        filename = session.get('file')
        project_name = session.get('name')
        type = session.get('type')
        df = pd.read_csv(filename)
        cols = df.columns
        cols_type = list(df.dtypes)
        try:
            x= (df[x_axis].astype(float)).to_list()
            y = (df[y_axis].astype(float)).to_list()
            x.append(x_axis)
            y.append(y_axis)



        except:
            x=None
            y=None



        return render_template('chart.html', cols=cols, cols_type=cols_type, x=x, y=y, len=len(cols),len1=len(x)-1,project_name=project_name,type=type)
    elif 'file' in session:
        filename = session.get('file')
        project_name = session.get('name')
        type = session.get('type')
        df = pd.read_csv(filename)
        cols = df.columns
        cols_type = list(df.dtypes)

        trigger = session.get('demo_trigger')
        return render_template('chart.html',cols=cols,cols_type=cols_type,x=None,y=None,len=len(cols),len1=0,trigger=trigger,project_name=project_name,type=type)
    else:
        return render_template('login.html')

#box_chart
@app.route('/box_chart',methods=['POST','GET'])
def box_chart():
    if 'file' in session:
        filename = session.get('file')
        project_name = session.get('name')
        type = session.get('type')
        df = pd.read_csv(filename)
        cols = df.columns
        cols_type = list(df.dtypes)
        #quantile calculations
        max_=[]
        q2=[]
        q3=[]
        min_=[]
        q4=[]
        x=[]
        for i in range(0,len(cols)):
            if cols_type[i]!='object' and math.isnan(max(df[cols[i]]))==False:

                max_.append(max(df[cols[i]]))
                min_.append(min(df[cols[i]]))
                q2.append( np.quantile(df[cols[i]], .25))
                q3.append(np.quantile(df[cols[i]], .50))
                q4.append(np.quantile(df[cols[i]], .75))
                x.append(cols[i])

        data={'max_':max_,'min_':min_,'q2':q2,'q3':q3,'q4':q4}

        trigger = session.get('demo_trigger')
        return render_template('box_chart.html',data=data,len=len(x),x=x,trigger=trigger,project_name=project_name,type=type)
    else:
        return render_template('login.html')
#Predictions
@app.route('/predict',methods=['POST','GET'])
def predict():
    pred_res = {}
    if 'file' in session and request.method == 'POST':
        filename = session.get('file')
        df = pd.read_csv(filename)
        cols = df.columns
        project_name = session.get('name')
        type = session.get('type')

        #form action
        target= request.form.getlist('target')

        #saving in session
        session['target'] = target

        split_type=request.form['split_type']


        #splitting the data
        x=df.drop(target,axis=1)
        y=df[target]

        # type--> regression
        if (type=='regression'):
            #choosing split type and predicting
            details={}
            if (split_type=='70_30_split'):
                X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.30, random_state=1)

                #linear/multi regression
                regression_model = LinearRegression()
                regression_model.fit(X_train, Y_train)
                accurracy=round(regression_model.score(X_test,Y_test)*100,3)
                pred_res["Linear_multiple_Regression"] = accurracy
                joblib.dump(regression_model, 'models/linear_multi_regression.pkl')#saving model

                details['Linear_multiple_Regression']={}
                details['Linear_multiple_Regression']['Coefficients'] = regression_model.coef_
                details['Linear_multiple_Regression']['Intercept'] = regression_model.intercept_
                Y_test = np.ravel(Y_test)
                pred = regression_model.predict(X_test)
                pred = np.ravel(pred)
                details['Linear_multiple_Regression']['Mean_squared_error'] = metrics.mean_squared_error(Y_test, pred)
                details['Linear_multiple_Regression']['Mean_absolute_error'] = metrics.mean_absolute_error(Y_test, pred)
#**********************************************************************************************************


                #polynomial regression (degree-2)
                poly = PolynomialFeatures(degree=2, interaction_only=True)
                X_train2 = poly.fit_transform(X_train)
                X_test2 = poly.fit_transform(X_test)
                poly_clf = linear_model.LinearRegression()
                poly_clf.fit(X_train2, Y_train)
                accurracy=round(poly_clf.score(X_test2, Y_test)*100,3)
                pred_res["Polynomial_Regression(d:2)"] = accurracy
                joblib.dump(poly_clf, 'models/polynomial_regression.pkl')  # saving model

                details['Polynomial_Regression(d:2)'] = {}
                details['Polynomial_Regression(d:2)']['Coefficients'] = poly_clf.coef_
                details['Polynomial_Regression(d:2)']['Intercept'] = poly_clf.intercept_
                Y_test = np.ravel(Y_test)
                pred = poly_clf.predict(X_test2)
                pred = np.ravel(pred)
                details['Polynomial_Regression(d:2)']['Mean_squared_error'] = metrics.mean_squared_error(Y_test, pred)
                details['Polynomial_Regression(d:2)']['Mean_absolute_error'] = metrics.mean_absolute_error(Y_test, pred)




#*******************************************************************************************************************************
                #Ridge Regression
                ridge = Ridge(alpha=.3)
                ridge.fit(X_train, Y_train)
                accurracy = round(ridge.score(X_test, Y_test) * 100, 3)
                pred_res["Ridge_Regression"] = accurracy
                joblib.dump(ridge, 'models/ridge_regression.pkl')  # saving model

                details['Ridge_Regression'] = {}
                details['Ridge_Regression']['Coefficients'] = ridge.coef_
                details['Ridge_Regression']['Intercept'] = ridge .intercept_
                Y_test = np.ravel(Y_test)
                pred = ridge.predict(X_test)
                pred = np.ravel(pred)
                details['Ridge_Regression']['Mean_squared_error'] = metrics.mean_squared_error(Y_test, pred)
                details['Ridge_Regression']['Mean_absolute_error'] = metrics.mean_absolute_error(Y_test, pred)
#**********************************************************************************************************
                #lasso Regression
                lasso = Lasso(alpha=0.1)
                lasso.fit(X_train, Y_train)
                accurracy = round(lasso.score(X_test, Y_test) * 100, 3)
                pred_res["Lasso_Regression"] = accurracy
                joblib.dump(lasso, 'models/lasso_regression.pkl')  # saving model

                details['Lasso_Regression'] = {}
                details['Lasso_Regression']['Coefficients'] = lasso.coef_
                details['Lasso_Regression']['Intercept'] = lasso.intercept_
                Y_test = np.ravel(Y_test)
                pred = ridge.predict(X_test)
                pred = np.ravel(pred)
                details['Lasso_Regression']['Mean_squared_error'] = metrics.mean_squared_error(Y_test, pred)
                details['Lasso_Regression']['Mean_absolute_error'] = metrics.mean_absolute_error(Y_test, pred)


                session['demo_trigger'] = True  #To view the demo model page
                trigger = session.get('demo_trigger')
                return render_template('prediction.html', cols=cols, type=type, pred_res=pred_res,split_type=split_type,details=details,trigger=trigger,project_name=project_name)
#**********************************************************************************************************************



            #k fold cross validation
            else:
                y=np.ravel(y)


                #linear/multi regression
                score=[]
                x_index=[]
                y_index=[]
                xt_index = []
                yt_index = []#variables to capture best test_train split
                regression_model = LinearRegression()
                for train_index, test_index in kf.split(x):
                    X_train, X_test = x.iloc[train_index, :], x.iloc[test_index, :]
                    y_train, y_test = y[train_index], y[test_index]
                    x_index.append(X_train)
                    y_index.append(y_train)
                    xt_index.append(X_test)
                    yt_index.append(y_test)
                    regression_model.fit(X_train, y_train)
                    acc = regression_model.score(X_test, y_test)
                    score.append(acc)
                regression_model.fit(x_index[score.index(max(score))], y_index[score.index(max(score))])
                pred_res["Linear_multiple_Regression"] = round(max(score)*100,2)
                joblib.dump(regression_model, 'models/linear_multi_regression.pkl')  # saving model

                details['Linear_multiple_Regression'] = {}
                details['Linear_multiple_Regression']['Coefficients'] = regression_model.coef_
                details['Linear_multiple_Regression']['Intercept'] = regression_model.intercept_
                Y_test = np.ravel(yt_index[score.index(max(score))])
                pred = regression_model.predict(xt_index[score.index(max(score))])
                pred = np.ravel(pred)
                details['Linear_multiple_Regression']['Mean_squared_error'] = metrics.mean_squared_error(Y_test, pred)
                details['Linear_multiple_Regression']['Mean_absolute_error'] = metrics.mean_absolute_error(Y_test, pred)
#***********************************************************************************************
                #Polynomial regression
                score = []
                x_index = []
                y_index = []
                xt_index = []
                yt_index = []  # variables to capture best test_train split
                poly_clf = linear_model.LinearRegression()
                for train_index, test_index in kf.split(x):
                    poly = PolynomialFeatures(degree=2, interaction_only=True)
                    X_train, X_test = x.iloc[train_index, :], x.iloc[test_index, :]
                    y_train, y_test = y[train_index], y[test_index]
                    X_train2 = poly.fit_transform(X_train)
                    X_test2 = poly.fit_transform(X_test)
                    x_index.append(X_train)
                    y_index.append(y_train)
                    xt_index.append(X_test)
                    yt_index.append(y_test)

                    poly_clf.fit(X_train2, y_train)
                    acc = poly_clf.score(X_test2, y_test)
                    score.append(acc)

                pred_res["Polynomial_Regression(d:2)"] = round(max(score) * 100, 2)
                poly_clf.fit(x_index[score.index(max(score))], y_index[score.index(max(score))])
                joblib.dump(poly_clf, 'models/polynomial_regression.pkl')  # saving model

                details['Polynomial_Regression(d:2)'] = {}
                details['Polynomial_Regression(d:2)']['Coefficients'] = poly_clf.coef_
                details['Polynomial_Regression(d:2)']['Intercept'] = poly_clf.intercept_
                Y_test = np.ravel(yt_index[score.index(max(score))])
                pred = poly_clf.predict(xt_index[score.index(max(score))])
                pred = np.ravel(pred)
                details['Polynomial_Regression(d:2)']['Mean_squared_error'] = metrics.mean_squared_error(Y_test, pred)
                details['Polynomial_Regression(d:2)']['Mean_absolute_error'] = metrics.mean_absolute_error(Y_test, pred)
#*************************************************************************************************************************************

                # ridge regression
                score = []
                x_index = []
                y_index = []
                xt_index = []
                yt_index = []# variables to capture best test_train split
                ridge = Ridge(alpha=.3)
                for train_index, test_index in kf.split(x):
                    X_train, X_test = x.iloc[train_index, :], x.iloc[test_index, :]
                    y_train, y_test = y[train_index], y[test_index]
                    x_index.append(X_train)
                    y_index.append(y_train)
                    xt_index.append(X_test)
                    yt_index.append(y_test)

                    ridge.fit(X_train, y_train)
                    acc = ridge.score(X_test, y_test)
                    score.append(acc)

                pred_res["Ridge_Regression"] = round(max(score)*100,2)
                ridge.fit(x_index[score.index(max(score))], y_index[score.index(max(score))])
                joblib.dump(ridge, 'models/ridge_regression.pkl')  # saving model

                details['Ridge_Regression'] = {}
                details['Ridge_Regression']['Coefficients'] = ridge.coef_
                details['Ridge_Regression']['Intercept'] = ridge.intercept_
                Y_test = np.ravel(yt_index[score.index(max(score))])
                pred = ridge.predict(xt_index[score.index(max(score))])
                pred = np.ravel(pred)
                details['Ridge_Regression']['Mean_squared_error'] = metrics.mean_squared_error(Y_test, pred)
                details['Ridge_Regression']['Mean_absolute_error'] = metrics.mean_absolute_error(Y_test, pred)



#*******************************************************************************************************************8
                # lasso regression
                score = []
                x_index = []
                y_index = []
                xt_index = []
                yt_index = []# variables to capture best test_train split
                lasso = Lasso(alpha=0.1)
                for train_index, test_index in kf.split(x):
                    X_train, X_test = x.iloc[train_index, :], x.iloc[test_index, :]
                    y_train, y_test = y[train_index], y[test_index]
                    x_index.append(X_train)
                    y_index.append(y_train)
                    xt_index.append(X_test)
                    yt_index.append(y_test)

                    lasso.fit(X_train, y_train)
                    acc = lasso.score(X_test, y_test)
                    score.append(acc)

                pred_res["Lasso_Regression"] = round(max(score) * 100, 2)
                lasso.fit(x_index[score.index(max(score))], y_index[score.index(max(score))])
                joblib.dump(lasso, 'models/lasso_regression.pkl')  # saving model

                details['Lasso_Regression'] = {}
                details['Lasso_Regression']['Coefficients'] = lasso.coef_
                details['Lasso_Regression']['Intercept'] = lasso.intercept_
                Y_test = np.ravel(yt_index[score.index(max(score))])
                pred = lasso.predict(xt_index[score.index(max(score))])
                pred = np.ravel(pred)
                details['Lasso_Regression']['Mean_squared_error'] = metrics.mean_squared_error(Y_test, pred)
                details['Lasso_Regression']['Mean_absolute_error'] = metrics.mean_absolute_error(Y_test, pred)

                session['demo_trigger'] = True  # To view the demo model page
                trigger = session.get('demo_trigger')
                return render_template('prediction.html', cols=cols, type=type, pred_res=pred_res,split_type=split_type,details=details,trigger=trigger,project_name=project_name)
#***************************************************************************************************************************************8
        #type-->classification
        else:
            details = {}
            if (split_type=='70_30_split'):
                X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.30, random_state=1)

                #logistic regression
                lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
                lr.fit(X_train, Y_train)
                y_predict = lr.predict(X_test)
                accuracy = metrics.accuracy_score(Y_test, y_predict)
                pred_res["Logistic_Regression"] = round(accuracy * 100, 3)
                joblib.dump(lr, 'models/logistic_regression.pkl')  # saving model

                details['Logistic_Regression'] = {}
                confusion_matrix=metrics.confusion_matrix(Y_test,y_predict)
                FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
                FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
                TP = np.diag(confusion_matrix)
                TN = confusion_matrix.sum() - (FP + FN + TP)
                details['Logistic_Regression']['Confusion_Matrix'] = metrics.confusion_matrix(Y_test, y_predict)
                details['Logistic_Regression']['True_positive']=TP
                details['Logistic_Regression']['True_negative'] = TN
                details['Logistic_Regression']['False_positive'] = FP
                details['Logistic_Regression']['False_negative'] = FN
                details['Logistic_Regression']['Sensitivity_OR_Recall']=TP/(TP+FN)
                recall=TP/(TP+FN)
                details['Logistic_Regression']['Specivity'] = TN/(TN+FP)
                details['Logistic_Regression']['Precision'] = TP / (TP + FP)
                precision=TP / (TP + FP)
                details['Logistic_Regression']['F1_score'] = 2*((precision*recall)/(precision+recall))



#********************************************************************************************************************************************
                #knn
                score=[]
                neigh=[]
                for i in range(1, 40):
                    neigh.append(i)
                    knn = KNeighborsClassifier(n_neighbors=i)
                    knn.fit(X_train, Y_train)
                    y_predict = knn.predict(X_test)
                    score.append(metrics.accuracy_score(Y_test, y_predict))
                knn = KNeighborsClassifier(n_neighbors=neigh[score.index(max(score))])
                knn.fit(X_train, Y_train)
                pred_res["K_nearest_neighbour_(best_of_40_neighbours)"] = round(max(score) * 100, 3)
                joblib.dump(knn, 'models/knn.pkl')  # saving model

                details['K_nearest_neighbour_(best_of_40_neighbours)'] = {}
                confusion_matrix = metrics.confusion_matrix(Y_test, y_predict)
                FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
                FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
                TP = np.diag(confusion_matrix)
                TN = confusion_matrix.sum() - (FP + FN + TP)
                details['K_nearest_neighbour_(best_of_40_neighbours)']['Confusion_Matrix'] = metrics.confusion_matrix(Y_test, y_predict)
                details['K_nearest_neighbour_(best_of_40_neighbours)']['True_positive'] = TP
                details['K_nearest_neighbour_(best_of_40_neighbours)']['True_negative'] = TN
                details['K_nearest_neighbour_(best_of_40_neighbours)']['False_positive'] = FP
                details['K_nearest_neighbour_(best_of_40_neighbours)']['False_negative'] = FN
                details['K_nearest_neighbour_(best_of_40_neighbours)']['Sensitivity_OR_Recall'] = TP / (TP + FN)
                recall = TP / (TP + FN)
                details['K_nearest_neighbour_(best_of_40_neighbours)']['Specivity'] = TN / (TN + FP)
                details['K_nearest_neighbour_(best_of_40_neighbours)']['Precision'] = TP / (TP + FP)
                precision = TP / (TP + FP)
                details['K_nearest_neighbour_(best_of_40_neighbours)']['F1_score'] = 2 * ((precision * recall) / (precision + recall))




                #decision tree
                clf = DecisionTreeClassifier(random_state=5)
                clf.fit(X_train, Y_train)
                y_predict = clf.predict(X_test)
                accuracy = metrics.accuracy_score(Y_test, y_predict)
                pred_res["Decision_tree"] = round(accuracy * 100, 3)
                joblib.dump(clf, 'models/decision_tree.pkl')  # saving model

                details['Decision_tree'] = {}
                confusion_matrix = metrics.confusion_matrix(Y_test, y_predict)
                FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
                FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
                TP = np.diag(confusion_matrix)
                TN = confusion_matrix.sum() - (FP + FN + TP)
                details['Decision_tree']['Confusion_Matrix'] = metrics.confusion_matrix(Y_test, y_predict)
                details['Decision_tree']['True_positive'] = TP
                details['Decision_tree']['True_negative'] = TN
                details['Decision_tree']['False_positive'] = FP
                details['Decision_tree']['False_negative'] = FN
                details['Decision_tree']['Sensitivity_OR_Recall'] = TP / (TP + FN)
                recall = TP / (TP + FN)
                details['Decision_tree']['Specivity'] = TN / (TN + FP)
                details['Decision_tree']['Precision'] = TP / (TP + FP)
                precision = TP / (TP + FP)
                details['Decision_tree']['F1_score'] = 2 * ((precision * recall) / (precision + recall))


                #Navie bayes
                model = GaussianNB()
                model.fit(X_train, Y_train)
                y_predict = model.predict(X_test)
                accuracy = metrics.accuracy_score(Y_test, y_predict)
                pred_res["Navie_Bayes"] = round(accuracy * 100, 3)
                joblib.dump(model, 'models/navie_bayes.pkl')  # saving model

                details['Navie_Bayes'] = {}
                confusion_matrix = metrics.confusion_matrix(Y_test, y_predict)
                FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
                FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
                TP = np.diag(confusion_matrix)
                TN = confusion_matrix.sum() - (FP + FN + TP)
                details['Navie_Bayes']['Confusion_Matrix'] = metrics.confusion_matrix(Y_test, y_predict)
                details['Navie_Bayes']['True_positive'] = TP
                details['Navie_Bayes']['True_negative'] = TN
                details['Navie_Bayes']['False_positive'] = FP
                details['Navie_Bayes']['False_negative'] = FN
                details['Navie_Bayes']['Sensitivity_OR_Recall'] = TP / (TP + FN)
                recall = TP / (TP + FN)
                details['Navie_Bayes']['Specivity'] = TN / (TN + FP)
                details['Navie_Bayes']['Precision'] = TP / (TP + FP)
                precision = TP / (TP + FP)
                details['Navie_Bayes']['F1_score'] = 2 * ((precision * recall) / (precision + recall))

                session['demo_trigger'] = True  # To view the demo model page
                trigger = session.get('demo_trigger')
                return render_template('prediction.html', cols=cols, type=type, pred_res=pred_res,split_type=split_type,details=details,trigger=trigger,project_name=project_name)
            else:
                y = np.ravel(y)
                # logistic regression
                score = []
                x_index = []
                y_index = []
                xt_index = []
                yt_index = []  # variables to capture best test_train split
                lr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
                for train_index, test_index in kf.split(x):
                    X_train, X_test = x.iloc[train_index, :], x.iloc[test_index, :]
                    y_train, y_test = y[train_index], y[test_index]
                    lr.fit(X_train, y_train)
                    x_index.append(X_train)
                    y_index.append(y_train)
                    xt_index.append(X_test)
                    yt_index.append(y_test)#append splited data
                    y_predict = lr.predict(X_test)
                    accuracy = metrics.accuracy_score(y_test, y_predict)
                    score.append(accuracy)
                pred_res["Logistic_Regression"] = round(max(score) * 100, 3)
                lr.fit(x_index[score.index(max(score))], y_index[score.index(max(score))])
                joblib.dump(lr, 'models/logistic_regression.pkl')  # saving model

                y_predict = lr.predict(xt_index[score.index(max(score))])
                details['Logistic_Regression'] = {}
                confusion_matrix = metrics.confusion_matrix(yt_index[score.index(max(score))], y_predict)
                FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
                FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
                TP = np.diag(confusion_matrix)
                TN = confusion_matrix.sum() - (FP + FN + TP)
                details['Logistic_Regression']['Confusion_Matrix'] = metrics.confusion_matrix(yt_index[score.index(max(score))], y_predict)
                details['Logistic_Regression']['True_positive'] = TP
                details['Logistic_Regression']['True_negative'] = TN
                details['Logistic_Regression']['False_positive'] = FP
                details['Logistic_Regression']['False_negative'] = FN
                details['Logistic_Regression']['Sensitivity_OR_Recall'] = TP / (TP + FN)
                recall = TP / (TP + FN)
                details['Logistic_Regression']['Specivity'] = TN / (TN + FP)
                details['Logistic_Regression']['Precision'] = TP / (TP + FP)
                precision = TP / (TP + FP)
                details['Logistic_Regression']['F1_score'] = 2 * ((precision * recall) / (precision + recall))

#**************************************************************************************************************************************


                # knn
                score = []
                neigh = []
                x_index = []
                y_index = []
                xt_index = []
                yt_index = []  # variables to capture best test_train split
                for train_index, test_index in kf.split(x):
                    X_train, X_test = x.iloc[train_index, :], x.iloc[test_index, :]
                    y_train, y_test = y[train_index], y[test_index]
                    temp_score = []
                    temp_neigh=[]
                    x_index.append(X_train)
                    y_index.append(y_train)  # append splited data
                    xt_index.append(X_test)
                    yt_index.append(y_test)
                    for i in range(1, 40):
                        neigh.append(i)
                        knn = KNeighborsClassifier(n_neighbors=i)
                        knn.fit(X_train, y_train)
                        y_predict = knn.predict(X_test)
                        temp_score.append(metrics.accuracy_score(y_test, y_predict))
                        temp_neigh.append(i)
                    score.append(max(temp_score))
                    neigh.append(temp_neigh[temp_score.index(max(temp_score))])
                pred_res["K_nearest_neighbour_(best_of_40_neighbours)"] = round(max(score) * 100, 3)
                knn = KNeighborsClassifier(n_neighbors=neigh[score.index(max(score))])
                knn.fit(x_index[score.index(max(score))], y_index[score.index(max(score))])
                joblib.dump(knn, 'models/knn.pkl')

                y_predict = lr.predict(xt_index[score.index(max(score))])
                details['K_nearest_neighbour_(best_of_40_neighbours)'] = {}
                confusion_matrix = metrics.confusion_matrix(yt_index[score.index(max(score))], y_predict)
                FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
                FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
                TP = np.diag(confusion_matrix)
                TN = confusion_matrix.sum() - (FP + FN + TP)
                details['K_nearest_neighbour_(best_of_40_neighbours)']['Confusion_Matrix'] = metrics.confusion_matrix(
                    yt_index[score.index(max(score))], y_predict)
                details['K_nearest_neighbour_(best_of_40_neighbours)']['True_positive'] = TP
                details['K_nearest_neighbour_(best_of_40_neighbours)']['True_negative'] = TN
                details['K_nearest_neighbour_(best_of_40_neighbours)']['False_positive'] = FP
                details['K_nearest_neighbour_(best_of_40_neighbours)']['False_negative'] = FN
                details['K_nearest_neighbour_(best_of_40_neighbours)']['Sensitivity_OR_Recall'] = TP / (TP + FN)
                recall = TP / (TP + FN)
                details['K_nearest_neighbour_(best_of_40_neighbours)']['Specivity'] = TN / (TN + FP)
                details['K_nearest_neighbour_(best_of_40_neighbours)']['Precision'] = TP / (TP + FP)
                precision = TP / (TP + FP)
                details['K_nearest_neighbour_(best_of_40_neighbours)']['F1_score'] = 2 * ((precision * recall) / (precision + recall))

#************************************************************************************************************************************************

                # decision tree
                score = []
                x_index = []
                y_index = []
                xt_index = []
                yt_index = []  # variables to capture best test_train split
                clf = DecisionTreeClassifier(random_state=5)
                for train_index, test_index in kf.split(x):
                    X_train, X_test = x.iloc[train_index, :], x.iloc[test_index, :]
                    y_train, y_test = y[train_index], y[test_index]
                    x_index.append(X_train)
                    y_index.append(y_train)  # append splited data
                    xt_index.append(X_test)
                    yt_index.append(y_test)
                    clf.fit(X_train, y_train)
                    y_predict = clf.predict(X_test)
                    accuracy = metrics.accuracy_score(y_test, y_predict)
                    score.append(accuracy)
                pred_res["Decision_tree"] = round(max(score) * 100, 3)
                clf.fit(x_index[score.index(max(score))], y_index[score.index(max(score))])
                joblib.dump(clf, 'models/decision_tree.pkl')  # saving model

                y_predict = lr.predict(xt_index[score.index(max(score))])
                details['Decision_tree'] = {}
                confusion_matrix = metrics.confusion_matrix(yt_index[score.index(max(score))], y_predict)
                FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
                FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
                TP = np.diag(confusion_matrix)
                TN = confusion_matrix.sum() - (FP + FN + TP)
                details['Decision_tree']['Confusion_Matrix'] = metrics.confusion_matrix(yt_index[score.index(max(score))], y_predict)
                details['Decision_tree']['True_positive'] = TP
                details['Decision_tree']['True_negative'] = TN
                details['Decision_tree']['False_positive'] = FP
                details['Decision_tree']['False_negative'] = FN
                details['Decision_tree']['Sensitivity_OR_Recall'] = TP / (TP + FN)
                recall = TP / (TP + FN)
                details['Decision_tree']['Specivity'] = TN / (TN + FP)
                details['Decision_tree']['Precision'] = TP / (TP + FP)
                precision = TP / (TP + FP)
                details['Decision_tree']['F1_score'] = 2 * ((precision * recall) / (precision + recall))

#******************************************************************************************************************************



                # Navie bayes
                score = []
                x_index = []
                y_index = []
                xt_index = []
                yt_index = []  # variables to capture best test_train split
                model = GaussianNB()
                for train_index, test_index in kf.split(x):
                    X_train, X_test = x.iloc[train_index, :], x.iloc[test_index, :]
                    y_train, y_test = y[train_index], y[test_index]
                    x_index.append(X_train)
                    y_index.append(y_train)  # append splited data
                    xt_index.append(X_test)
                    yt_index.append(y_test)
                    model.fit(X_train, y_train)
                    y_predict = model.predict(X_test)
                    accuracy = metrics.accuracy_score(y_test, y_predict)
                    score.append(accuracy)
                pred_res["Navie_Bayes"] = round(max(score) * 100, 3)
                model.fit(x_index[score.index(max(score))], y_index[score.index(max(score))])
                joblib.dump(model, 'models/navie_bayes.pkl')  # saving model



                y_predict = lr.predict(xt_index[score.index(max(score))])
                details['Navie_Bayes'] = {}
                confusion_matrix = metrics.confusion_matrix(yt_index[score.index(max(score))], y_predict)
                FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
                FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
                TP = np.diag(confusion_matrix)
                TN = confusion_matrix.sum() - (FP + FN + TP)
                details['Navie_Bayes']['Confusion_Matrix'] = metrics.confusion_matrix(
                    yt_index[score.index(max(score))], y_predict)
                details['Navie_Bayes']['True_positive'] = TP
                details['Navie_Bayes']['True_negative'] = TN
                details['Navie_Bayes']['False_positive'] = FP
                details['Navie_Bayes']['False_negative'] = FN
                details['Navie_Bayes']['Sensitivity_OR_Recall'] = TP / (TP + FN)
                recall = TP / (TP + FN)
                details['Navie_Bayes']['Specivity'] = TN / (TN + FP)
                details['Navie_Bayes']['Precision'] = TP / (TP + FP)
                precision = TP / (TP + FP)
                details['Navie_Bayes']['F1_score'] = 2 * ((precision * recall) / (precision + recall))


                session['demo_trigger'] = True  # To view the demo model page
                trigger = session.get('demo_trigger')
                return render_template('prediction.html', cols=cols, type=type, pred_res=pred_res,split_type=split_type,details=details,trigger=trigger,project_name=project_name)

#************************************************************************************************************************

    elif 'file' in session:
        filename = session.get('file')
        df = pd.read_csv(filename)
        cols = df.columns
        project_name = session.get('name')
        type = session.get('type')
        trigger = session.get('demo_trigger')
        return render_template('prediction.html', cols=cols, type=type,pred_res=pred_res,trigger=trigger,project_name=project_name)




    else:
        return render_template('login.html')
@app.route('/demo',methods=['POST','GET'])
def demo():
    filename = session.get('file')
    project_name = session.get('name')
    df = pd.read_csv(filename)
    target = session.get('target')
    type = session.get('type')
    x_cols = df.drop(target, axis=1)
    if 'file' in session and request.method == 'POST':
        res=[]
        for i in x_cols:
            a=int(request.form[i])
            res.append(a)
        res=[np.array(res)]
        algo=request.form['algo']

        if algo=='linear':
            model = joblib.load('models/linear_multi_regression.pkl')
            output=model.predict(res)

        elif algo=='poly':
            model = joblib.load('models/polynomial_regression.pkl')
            output = model.predict(res)

        elif algo=='ridge':
            model = joblib.load('models/ridge_regression.pkl')
            output = model.predict(res)

        elif algo=='lasso':
            model = joblib.load('models/lasso_regression.pkl')
            output = model.predict(res)

        elif algo=='logistic':
            model = joblib.load('models/logistic_regression.pkl')
            output = model.predict(res)
            print(2)

        elif algo=='knn':
            model = joblib.load('models/knn.pkl')
            output = model.predict(res)
        elif algo=='decision':
            model = joblib.load('models/decision_tree.pkl')
            output = model.predict(res)
        else:
            model = joblib.load('models/navie_bayes.pkl')
            output = model.predict(res)

        trigger = session.get('demo_trigger')
        return render_template('explore.html', x_cols=x_cols, type=type, trigger=trigger, algo=algo, project_name=project_name,output=output)

    elif 'file' in session:
        filename = session.get('file')
        project_name = session.get('name')
        df = pd.read_csv(filename)

        type = session.get('type')
        algo=request.args.get('a')


        trigger = session.get('demo_trigger')
        return render_template('explore.html', x_cols=x_cols, type=type, trigger=trigger,algo=algo,project_name=project_name)


    else:
        return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))
#
if __name__ == "__main__":
    app.run(debug=True)