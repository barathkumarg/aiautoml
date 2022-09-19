from Code import *

# description display
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



