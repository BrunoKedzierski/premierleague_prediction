from fastapi import FastAPI
import pickle
from football_game import Football_Game
import pandas as pd
from fastapi import FastAPI
import uvicorn



## Klasa pomocnicza - wczytywanie modeli
class ModelHandler:
    def __init__(self):
        self.loaded_model = None
        self.loaded_encoder = None
        self.model_artifact =  None

    def load_models(self):
        self.loaded_model = pickle.load(open('model\\xgboost_model.pkl', 'rb'))
        self.loaded_encoder =pickle.load(open('model\\encoder.pkl', 'rb'))

app = FastAPI()
model_handler = ModelHandler()


#Zaladuj modele na start
@app.on_event("startup")
def load_models():
    model_handler.load_models()


@app.get("/")
def read_root():
    return {"Hello": "World"}

#Informacje o modelu
@app.get("/modelinfo")
def show_model_info():
    return {"model_type": type(model_handler.loaded_model).__name__}


#Predykcja
@app.post("/predict/match")
def predict_client(data: Football_Game):
    
    data = data.dict()

    print(data)
    dframe = pd.DataFrame(data, index=[0])
    print(dframe)
    encoded_columns = model_handler.loaded_encoder.transform(dframe[['HomeTeam', 'AwayTeam', 'HTR']])

    
    feature_names = model_handler.loaded_encoder.get_feature_names_out(input_features=['HomeTeam', 'AwayTeam', 'HTR'])

    
    data_encoded = pd.DataFrame(encoded_columns, columns=feature_names)

  
    missing_columns = set(model_handler.loaded_encoder.get_feature_names_out()) - set(data_encoded.columns)
    for col in missing_columns:
        data_encoded[col] = 0.0

    data_encoded = pd.concat([dframe.drop(columns=['HomeTeam', 'AwayTeam', 'HTR']), data_encoded], axis=1)

    print("After concat")
    

    prediction = model_handler.loaded_model.predict(data_encoded)
    print(prediction)
    prob = model_handler.loaded_model.predict_proba(data_encoded)
    print(prob)

    print(prediction.shape)
    print(prob.shape)
    return {
    'prediction': prediction.tolist()[0],
    'probabilityHome': prob.flatten().tolist()[0],
    'probabilityDraw': prob.flatten().tolist()[1],
    'probabilityAway': prob.flatten().tolist()[2],
}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8001)
