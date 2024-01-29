from pydantic import BaseModel

class Football_Game(BaseModel):
  HomeTeam: str
  AwayTeam: str
  H_Ranking_Prior_Season: int
  A_Ranking_Prior_Season: int
  HTHG: int
  HTAG: int
  HTR: str
  HS: int
  AS: int
  B365H: float
  B365D: float
  B365A: float
  
