from Code import *

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
