from app import *
dict_ = {"Linear_multiple_Regression":"linear_multi_regression.pkl",

"Ridge_Regression" : "ridge_regression.pkl",
"Lasso_Regression" : "lasso_regression.pkl",
"Polynomial_Regression(d:2)":"polynomial_regression.pkl",

"Logistic_Regression":"logistic_regression.pkl",
"K_nearest_neighbour_(best_of_40_neighbours)":"knn.pkl",
"Decision_tree":"decision_tree.pkl",
"Navie_Bayes":"navie_bayes.pkl"

}

@app.route("/import_pickle",methods=["POST","GET"])
def import_pickle():
    id = request.args.get("id")
    project = session.get("name")
    print(id)
    return send_file(filename_or_fp= "../models/{}".format(dict_[id]),
        as_attachment=True, attachment_filename=f"{project}-{dict_[id]}")
