import streamlit as st
import pandas as pd
import pickle
import streamlit.components.v1 as components



components.html("""
          <h1 style="color: white; font-size: 40px; background-color: blue; border-radius:5px"><center>Predicting Stores Sales</center></h1>
               
                 <p style= "font-size:20px;"><center>To predict store sales, please enter the fields below.</center></p>
 """)

 
store_nbr = st.slider("store_nbr",0,54)
family = st.selectbox("products", ['AUTOMOTIVE', 'CLEANING', 'BEAUTY', 'FOODS', 'STATIONERY',
                                                           'CELEBRATION', 'GROCERY', 'HARDWARE', 'HOME', 'LADIESWEAR',
                                                           'LAWN AND GARDEN', 'CLOTHING', 'LIQUOR,WINE,BEER', 'PET SUPPLIES'])
onpromotion = st.number_input("onpromotion", min_value=0.0, max_value=800.0)
transactions = st.number_input("Number of Transactions", min_value=0.0, max_value=10000.0)
holiday_type = st.selectbox("holiday_type", ['Holiday', 'Additional', 'Transfer', 'Event', 'Bridge'])
oil_price = st.number_input("oil_price",step=1.0, min_value=0.0, max_value=200.0)
city = st.selectbox("state", ['Pichincha', 'Cotopaxi', 'Chimborazo', 'Imbabura',
                                                     'Santo Domingo de los Tsachilas', 'Bolivar', 'Pastaza',
                                                     'Tungurahua', 'Guayas', 'Santa Elena', 'Los Rios', 'Azuay', 'Loja',
                                                     'El Oro', 'Esmeraldas', 'Manabi'])
cluster= st.slider("cluster",step=1.0, min_value=0.0, max_value=17.0 )
day = st.slider("day",1,31)
year = st.slider("year",step=1.0, min_value=0.0, max_value=12.0 )
month = st.slider("month",1,12)
dayofmonth = st.slider("dayofmonth",1,31)
dayofweek = st.slider("dayofweek,0=Sun and 6=Sat",step=1, min_value=1, max_value=6)

if st.button("Predict"):
    df = pd.DataFrame({"store_nbr":[store_nbr],
                       "family":[family],
                       "onpromotion":[onpromotion],
                       "transactions":[transactions],
                       "holiday_type":[holiday_type],
                       "oil_price":[oil_price],
                       "city":[city],
                       "cluster":[cluster],
                       "day":[day],
                       "year":[year],
                       "month":[month],
                       "dayofmonth":[dayofmonth],
                       "dayofweek":[dayofweek],})
    
    with open("gboost_model.pkl", "rb") as f:
      model = pickle.load(f)

      output = model.predict(df)
      
      st.write("The Predicted Sells Is:",    output)
