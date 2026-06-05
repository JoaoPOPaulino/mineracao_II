from pathlib import Path

import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(
    title="API Regras de Associação"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "models" / "regras_associacao.csv"

regras = pd.read_csv(CSV_PATH)


@app.get("/")
def home():
    return FileResponse(
        BASE_DIR / "frontend" / "index.html"
    )


@app.get("/regras")
def listar_regras():

    top = (
        regras
        .sort_values("lift", ascending=False)
        .head(50)
    )

    return top.to_dict(orient="records")