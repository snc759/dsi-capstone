import streamlit as st
import pickle

from yelp_processing import get_recs
#-------------------------------------------------------------------------------------------

st.title('Restaurant Recommendations')
st.markdown('**Tuned to Your Desired Experience**')

page = st.selectbox("Navigate", ('About', 'Get Restaurant Recommendations'))

if page == 'Get Restaurant Recommendations': 

    st.write('CHOOSE YOUR FEATURES:')
    
    st.markdown("""
    <style>
    .big-font {
        font-size:18px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    
#     st.markdown('<p class="big-font">Kid-Friendly</p>', unsafe_allow_html=True)
    GoodForKids = st.radio('Kid-Friendly', ['Yes', 'No'])
#     GoodForKids = st.radio('', ['Yes', 'No'])
    GoodForGroups = st.radio('Group-Friendly', ['Yes', 'No'])
    OutdoorSeating = st.radio('Outdoor Seating', ['Yes', 'No'])
    Reservations = st.radio('Takes Reservations', ['Yes', 'No'])
    HasAlcohol = st.radio('Includes Bar/Wine/Alcohol', ['Yes', 'No'])
    TableService = st.radio('Table Service', ['Yes', 'No'])
    MealType = st.radio('What Type of Meal Does the Restaurant Serve?', ['Lunch', 'Dinner', 'Both', 'Other'])
    
    feature_list = [GoodForKids, GoodForGroups, OutdoorSeating, Reservations, HasAlcohol, TableService, MealType]
    
    user_text = st.text_area('Describe the restaurant experience you want: ')
    
    start_recommender = st.button('Submit')
    
    if user_text != '' and start_recommender:
        
        amb_preds, final_recs_df = get_recs(user_text, feature_list)
        final_recs_df.index += 1
        st.write('')
        st.write(f'Based on your description, you are looking for a `{amb_preds}` experience.')
        st.write('')
        st.write('')
        st.markdown('**Here are your restaurant recommendations:** ')
        st.table(final_recs_df)  
        
    elif user_text == '' and start_recommender: 
        st.error('Please describe the restaurant experience you want above.')
    
