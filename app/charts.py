from app import *
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