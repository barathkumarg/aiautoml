from Code import *
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))