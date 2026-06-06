from pathlib import Path

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
    """
    Normaliza a coluna de antecedentes/consequentes para o formato
    'Item A, Item B', independentemente de como foi salva no CSV.

    Trata os seguintes formatos possíveis:
      - Já correto:           'Cálculo 1, Física'
      - frozenset como str:   "frozenset({'Cálculo 1', 'Física'})"
      - Com aspas extras:     "{'Cálculo 1', 'Física'}"
    """
    import re

    def _normalizar(valor: str) -> str:
        s = str(valor).strip()

        # Detecta se ainda tem formato frozenset ou set
        if s.startswith("frozenset(") or s.startswith("{"):
            # Extrai tudo entre a primeira { e a última }
            m = re.search(r"\{(.+)\}", s, re.DOTALL)
            if m:
                conteudo = m.group(1)
                # Separa os itens (podem estar entre aspas simples ou duplas)
                itens = re.findall(r"['\"](.+?)['\"]", conteudo)
                return ", ".join(sorted(itens))

        return s

    return serie.apply(_normalizar)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega os dados na inicialização e libera no encerramento."""
    global regras
    if CSV_PATH.exists():
        regras = pd.read_csv(CSV_PATH)
        # Normaliza as colunas independentemente do formato salvo no CSV
        regras["antecedents"] = _limpar_coluna_itens(regras["antecedents"])
        regras["consequents"] = _limpar_coluna_itens(regras["consequents"])
        print(f"✅ {len(regras)} regras carregadas de {CSV_PATH}")
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
    """
    Retorna regras filtradas e ordenadas por lift.
    Parâmetros:
        min_lift: lift mínimo (padrão 1.0)
        min_confidence: confiança mínima (padrão 0.0)
        min_support: suporte mínimo (padrão 0.0)
        item: filtrar regras que contenham este item
        limit: número máximo de regras retornadas (padrão 50)
    """
    if regras.empty:
        return JSONResponse(
            status_code=503,
            content={"detail": "Dados ainda não carregados"},
        )

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