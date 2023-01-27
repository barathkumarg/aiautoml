from app import *
import pickle

@app.route("/test_pickle",methods=["GET","POST"])
def test_pickle():

    if request.method == 'POST':
        pickle_file = request.files['pickle']
        test_file = request.files['test']
        pickle_file.save(secure_filename("pickle_file.pkl"))
        test_file.save(secure_filename("test_file.csv"))

        #read csv to dataframe
        # try:
        output = pd.read_csv(r"C:\Users\barathkumar\Desktop\automl\autoaiml\test_file.csv")

        #logic to fetch the result
        with open(r"C:\Users\barathkumar\Desktop\automl\autoaiml\pickle_file.pkl", "rb") as file_handle:
            loaded_model = joblib.load(file_handle)
        result = loaded_model.predict(output)


        #fetching the value to display in html
        output['output'] = pd.Series(result)
        #saving file
        output.to_csv(r"C:\Users\barathkumar\Desktop\automl\autoaiml\test_file.csv",index=False)
        cols = output.columns  # csv columns
  
        return render_template("test_pickle.html",key=1,cols=cols,data=output,len1=len(output))
        # except Exception as ex:
        #     print(ex)
        #     return render_template("error.html")

    else:
        return render_template("test_pickle.html",key=0)
        