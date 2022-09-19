from Code import *

@app.route('/download_table',methods=['POST','GET'])
def download_table():
    filename = session.get('file')
    df = pd.read_csv(filename)
    return Response(df.to_csv(), mimetype="text/csv",
                    headers={"Content-disposition": "attachment; filename=result.csv"})

@app.route('/download_pred_table',methods=['POST','GET'])
def download_pred_table():

    df = pd.read_csv("test_file.csv")
    return Response(df.to_csv(), mimetype="text/csv",
                    headers={"Content-disposition": "attachment; filename=result.csv"})