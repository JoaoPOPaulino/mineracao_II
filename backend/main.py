from pathlib import Path
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "models" / "regras_associacao.csv"

regras: pd.DataFrame = pd.DataFrame()


def _limpar_coluna_itens(serie: pd.Series) -> pd.Series:
    """Limpa antecedentes/consequentes para exibição amigável."""
    def _normalizar(valor):
        s = str(valor).strip()
        # Remove possíveis frozenset / set / aspas extras
        s = s.replace("frozenset({", "").replace("})", "").replace("{", "").replace("}", "")
        s = s.replace("'", "").replace('"', "").strip()
        # Se tiver vírgula, ordena os itens
        if "," in s:
            itens = [x.strip() for x in s.split(",") if x.strip()]
            return ", ".join(sorted(itens))
        return s

    return serie.apply(_normalizar)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega os dados na inicialização e libera no encerramento."""
    global regras
    if CSV_PATH.exists():
        regras = pd.read_csv(CSV_PATH)
        regras["antecedents"] = _limpar_coluna_itens(regras["antecedents"])
        regras["consequents"] = _limpar_coluna_itens(regras["consequents"])
        print(f"✅ {len(regras)} regras carregadas e normalizadas.")
    else:
        print(f"⚠️  Arquivo não encontrado: {CSV_PATH}")
    yield


app = FastAPI(
    title="API Regras de Associação",
    description="Backend para dashboard de regras de associação com ECLAT",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def home():
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend não encontrado")
    return FileResponse(str(index))


@app.get("/regras")
def listar_regras(
    min_lift: float = 1.0,
    min_confidence: float = 0.0,
    min_support: float = 0.0,
    item: str = "",
    limit: int = 50,
):
    if regras.empty:
        return JSONResponse(status_code=503, content={"detail": "Dados ainda não carregados"})

    df = regras.copy()

    df = df[
        (df["lift"] >= min_lift)
        & (df["confidence"] >= min_confidence)
        & (df["support"] >= min_support)
    ]

    if item.strip():
        item_lower = item.strip().lower()
        mask = df["antecedents"].str.lower().str.contains(item_lower) | \
               df["consequents"].str.lower().str.contains(item_lower)
        df = df[mask]

    df = df.sort_values("lift", ascending=False).head(limit)

    # CORREÇÃO: Tratar Infinity e NaN antes de converter para JSON
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.astype(object).where(pd.notna(df), None)

    return df.to_dict(orient="records")


@app.get("/stats")
def estatisticas():
    """Retorna estatísticas gerais sobre as regras."""
    if regras.empty:
        return JSONResponse(
            status_code=503,
            content={"detail": "Dados ainda não carregados"},
        )

    return {
        "total_regras": int(len(regras)),
        "lift_medio": round(float(regras["lift"].mean()), 4),
        "lift_max": round(float(regras["lift"].max()), 4),
        "confianca_media": round(float(regras["confidence"].mean()), 4),
        "suporte_medio": round(float(regras["support"].mean()), 4),
        "itens_unicos": int(
            pd.concat([regras["antecedents"], regras["consequents"]])
            .str.split(", ")
            .explode()
            .nunique()
        ),
    }


@app.get("/itens")
def listar_itens():
    """Retorna todos os itens únicos presentes nas regras."""
    if regras.empty:
        return []

    itens = (
        pd.concat([regras["antecedents"], regras["consequents"]])
        .str.split(", ")
        .explode()
        .str.strip()
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(itens)