#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask
import joblib


# In[3]:


app = Flask(__name__)


# In[4]:


from flask import request, render_template
from keras.models import load_model

@app.route("/",methods=["GET", "POST"])
def index():
    if request.method=="POST":
        income = float(request.form.get("income"))
        age = float(request.form.get("age"))
        loan = float(request.form.get("loan"))
        print(income, age, loan)
        model_LogReg = joblib.load("LogisticRegression.jl")
        model_CART = joblib.load("CART.jl")
        model_RandomForest = joblib.load("RandomForest.jl")
        model_XGBoost = joblib.load("XGBoost.jl")
        model_MLP = joblib.load("MLP.jl")
        pred_LogReg = model_LogReg.predict([[income, age, loan]])
        pred_CART = model_CART.predict([[income, age, loan]])
        pred_RandomForest = model_RandomForest.predict([[income, age, loan]])
        pred_XGBoost = model_XGBoost.predict([[income, age, loan]])
        pred_MLP = model_MLP.predict([[income, age, loan]])
        print(pred_LogReg, pred_CART, pred_RandomForest, pred_XGBoost, pred_MLP)
        s = "Is the customer predicted to default?: " + "Logistic Regression: " + str(pred_LogReg) + " CART: " + str(pred_CART) + " RandomForest: " + str(pred_RandomForest) + " XGBoost: " + str(pred_XGBoost) + " MLP: " + str(pred_MLP)
        return(render_template("index.html", result=s))
    else:
        return(render_template("index.html", result="Error"))


# In[ ]:


app.run()


# In[ ]:




