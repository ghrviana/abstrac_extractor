"""
Gerador dinâmico de modelos Pydantic.

Cada aluno define seus próprios campos de extração pela interface (nome,
tipo, descrição e valor padrão). Este módulo transforma essa lista de
dicionários em uma classe Pydantic real, criada em tempo de execução com
`pydantic.create_model()` — é isso que permite que o app funcione para
qualquer conjunto de campos, sem programação por parte do aluno.
"""

import re
import unicodedata
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

# Mapeamento do tipo escolhido pelo aluno na interface para o tipo Python
# usado no modelo Pydantic. Campos do tipo "lista" são guardados como string
# (valores separados por vírgula), pois é assim que o LLM deve responder.
MAPA_TIPOS = {
    "texto": Optional[str],
    "número": Optional[float],
    "lista": Optional[str],
}


def sanitize_field_name(nome: str) -> str:
    """Converte um nome digitado pelo aluno (ex: 'Proteína Estudada') em um
    identificador válido para o modelo Pydantic e a chave do JSON
    (ex: 'proteina_estudada'): sem acentos, sem espaços, em minúsculas."""
    sem_acento = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode("ascii")
    sem_acento = sem_acento.strip().lower()
    return re.sub(r"\s+", "_", sem_acento)


def resolve_default_value(tipo: str, valor_padrao: str):
    """Converte o valor padrão (sempre digitado como texto na interface)
    para o tipo correto quando o campo é numérico. Também é usado fora deste
    módulo (em app.py) para comparar os valores extraídos com o valor padrão
    realmente armazenado no modelo, já no tipo certo."""
    if tipo == "número":
        try:
            return float(valor_padrao)
        except (TypeError, ValueError):
            return None
    return valor_padrao


def build_extraction_model(fields: list[dict]) -> type[BaseModel]:
    """Cria dinamicamente um modelo Pydantic a partir da lista de campos
    definida pelo aluno. Cada item de `fields` deve ter as chaves:
    'nome', 'tipo' ('texto'/'número'/'lista'), 'descricao' e 'valor_padrao'."""
    definicoes_dos_campos = {}
    for campo in fields:
        tipo_python = MAPA_TIPOS.get(campo["tipo"], Optional[str])
        valor_default = resolve_default_value(campo["tipo"], campo["valor_padrao"])
        definicoes_dos_campos[campo["nome"]] = (
            tipo_python,
            Field(default=valor_default, description=campo["descricao"]),
        )

    return create_model(
        "ExtractionModel",
        __config__=ConfigDict(populate_by_name=True),
        **definicoes_dos_campos,
    )


def generate_model_source(fields: list[dict]) -> str:
    """Monta uma representação em texto do modelo Pydantic gerado, para fins
    didáticos (exibida no painel 'O que está acontecendo nos bastidores?').
    Como `create_model()` cria a classe em tempo de execução, ela não possui
    um arquivo-fonte real para `inspect.getsource()` — por isso geramos esse
    texto manualmente, no mesmo formato que o aluno veria se escrevesse a
    classe à mão."""
    linhas = [
        "from typing import Optional",
        "from pydantic import BaseModel, ConfigDict, Field",
        "",
        "",
        "class ExtractionModel(BaseModel):",
        "    model_config = ConfigDict(populate_by_name=True)",
        "",
    ]

    nomes_dos_tipos = {"texto": "Optional[str]", "número": "Optional[float]", "lista": "Optional[str]"}

    for campo in fields:
        tipo_anotado = nomes_dos_tipos.get(campo["tipo"], "Optional[str]")
        valor_default = resolve_default_value(campo["tipo"], campo["valor_padrao"])
        default_repr = repr(valor_default)
        descricao_repr = repr(campo["descricao"])
        linhas.append(
            f"    {campo['nome']}: {tipo_anotado} = Field(default={default_repr}, description={descricao_repr})"
        )

    return "\n".join(linhas)
