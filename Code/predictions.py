from Code import *
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