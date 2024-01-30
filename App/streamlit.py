import streamlit as st
import base64
import pandas as pd
import pickle
from football_game import Football_Game
import requests
import matplotlib.pyplot as plt

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)






def main():



    set_background("background.png")

    st.title("Premier League Game Prediction")

   
    teams = ['Crystal Palace', 'Fulham', 'Bournemouth', 'Leeds', 'Newcastle',
       'Tottenham', 'Everton', 'Leicester', 'Man United', 'West Ham',
       'Aston Villa', 'Arsenal', 'Brighton', 'Man City', 'Southampton',
       'Wolves', 'Brentford', "Nott'm Forest", 'Chelsea', 'Liverpool']
    

    col1, col2, col3, col4, col5 = st.columns([6, 2, 4,2, 6])

    with col1:
        st.write("#### [Home Team:")
        team_h = st.selectbox("Team 1", teams, key="team1")  # Replace with actual team names
        half_goals_h = st.number_input("Half Time Goals", min_value=0, step=1, value=0, key="half_goals_h")
        half_shots_h = st.number_input("Half Time Shots", min_value=0, step=1, value=0, key="half_shots_h")
        ranking_prev_h = st.number_input("Ranking Prior Season", min_value=1, step=1, max_value= 18, value=1, key="ranking_prev_h")
        win_odds_h = st.number_input("Winning odds", min_value=1.0, step=0.1 , value=1.0, key="win_odds_h")

    with col2:
        pass

    with col3:
        st.write("")  # Empty space for VS text
        st.write("### VS")
        draw_odds = st.number_input("Draw odds", min_value=1.0, step=0.1 , value=1.0, key="draw_odds")
        half_time_res = st.selectbox("Half Time winner", ["H", "A", "D"], key="half_time_res ")

    with col4:
        pass    

    with col5:
        st.write("#### Away Team:")
        team_a = st.selectbox("Select team", teams, key="team2")  # Replace with actual team names
        half_goals_a = st.number_input("Half Time Goals", min_value=0, step=1, value=0, key="half_goals_a")
        half_shots_a = st.number_input("Half Time Shots", min_value=0, step=1, value=0, key="half_shots_a")
        ranking_prev_a = st.number_input("Ranking Prior Season", min_value=1, step=1, value=1, max_value= 18, key="ranking_prev_a")
        win_odds_a = st.number_input("Winning odds", step=0.1, min_value=1.0 , value=1.0, key="win_odds_a")



    if st.button("Predict", key="predict_button"):

        pred_request = Football_Game(HomeTeam=team_h, AwayTeam=team_a, HTAG=half_goals_a, HTHG=half_goals_h,
                                     HS=half_shots_h, AS=half_shots_a, A_Ranking_Prior_Season=ranking_prev_a, H_Ranking_Prior_Season= ranking_prev_h,
                                     B365A=win_odds_a, B365H=win_odds_h, B365D=draw_odds, HTR=half_time_res)

        prediction_response = requests.post('http://localhost:8001/predict/match', json=pred_request.dict()).json()
        winner =  prediction_response['prediction']
        st.write(winner)
        if(winner == 0):
            winner = "Home Team Wins!"
        elif(winner == 1):
            winner = "It's a Draw!"
        else:
            winner ="Away team Wins!" 
        st.write(winner)

        st.bar_chart(pd.DataFrame(
                        {
                            "Side": ['Home', 'Draw', 'Away'] ,
                            'Probabilty': [prediction_response['probabilityHome'], prediction_response['probabilityDraw'], prediction_response['probabilityAway']]
                              }  ), x="Side", y="Probabilty" )




      
                   

if __name__ == "__main__":
    main()