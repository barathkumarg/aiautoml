from Code import *

@app.route('/',methods=['POST','GET'])
def login():
    if request.method == 'POST':
        project_name=request.form['pname']
        type=request.form['type']
        file=request.files['file']


        file.save(secure_filename("input.csv"))
        session['name']=project_name
        session['type']=type
        session['file']="input.csv"
        session['demo_trigger'] = False  #To view the demo model page
        return redirect(url_for('home'))


    else:
        return render_template('login.html')
