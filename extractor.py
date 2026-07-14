"""
Extração estruturada de abstracts usando a API da DeepSeek + Pydantic.

Fluxo de cada chamada: montamos um system prompt explicando a tarefa e o
formato esperado -> chamamos a API em modo JSON -> validamos a resposta com
o modelo Pydantic dinâmico -> devolvemos um dicionário simples para a UI.

Se qualquer coisa falhar (chave inválida, limite de requisições, timeout,
etc.), a função NUNCA propaga a exceção: ela preenche o abstract com o valor
padrão global e descreve o erro na chave especial "_erro", para que a
interface possa avisar o usuário sem travar o processamento dos demais
abstracts do lote.
"""

import json

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from model_builder import resolve_default_value

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODELO_PADRAO = "deepseek-v4-flash"

# Os modelos da família V4 da DeepSeek vêm com o modo "thinking" (raciocínio
# estendido) habilitado por padrão. Para extração estruturada queremos
# respostas rápidas e determinísticas, então desativamos explicitamente —
# com "thinking" ligado o parâmetro `temperature` é ignorado silenciosamente.
THINKING_DESLIGADO = {"thinking": {"type": "disabled"}}


def build_system_prompt(fields: list[dict], model_class) -> str:
    """Monta o prompt em português que explica ao modelo o que extrair e em
    qual formato responder. A documentação da DeepSeek recomenda incluir um
    exemplo concreto do JSON esperado (além de instruir a responder só em
    JSON), por isso geramos um exemplo a partir dos próprios campos do
    aluno. Cada campo tem seu próprio valor padrão (definido pelo aluno na
    interface) para quando a informação não aparecer no texto — esse valor é
    diferente do valor padrão "global", que só é usado internamente quando a
    chamada à API falha por completo."""
    descricao_dos_campos = "\n".join(
        f'- "{campo["nome"]}" (tipo: {campo["tipo"]}, valor padrão se não encontrado: "{campo["valor_padrao"]}"): {campo["descricao"]}'
        for campo in fields
    )

    exemplo_json = {}
    for campo in fields:
        if campo["tipo"] == "lista":
            exemplo_json[campo["nome"]] = "valor1, valor2, valor3"
        elif campo["tipo"] == "número":
            exemplo_json[campo["nome"]] = 0.0
        else:
            exemplo_json[campo["nome"]] = "valor extraído do texto"

    return f"""Você é um assistente especializado em extrair informações estruturadas de abstracts científicos.

Sua tarefa é ler o abstract enviado pelo usuário e extrair os seguintes campos:
{descricao_dos_campos}

Regras importantes:
- Para campos do tipo "lista", retorne todos os valores encontrados separados por vírgula em uma única string (ex: "Arabidopsis, Nicotiana").
- Para campos do tipo "número", retorne apenas o valor numérico (sem unidades ou texto).
- Quando uma informação não puder ser encontrada no abstract, use o valor padrão indicado entre parênteses para aquele campo específico (não invente outro valor).
- Responda SOMENTE com um objeto JSON válido, sem nenhum texto antes ou depois, seguindo exatamente este formato de exemplo:

{json.dumps(exemplo_json, ensure_ascii=False, indent=2)}
"""


def _mensagem_amigavel_para_erro(erro: Exception) -> str:
    """Traduz exceções técnicas da API em mensagens amigáveis em português,
    sem expor stack traces ao usuário final (regra de negócio do projeto)."""
    if isinstance(erro, AuthenticationError):
        return "Chave de API inválida ou não autorizada. Verifique a chave fornecida pelo professor."
    if isinstance(erro, RateLimitError):
        return "Limite de requisições da API atingido. Aguarde um pouco e tente novamente."
    if isinstance(erro, APITimeoutError):
        return "Tempo limite excedido ao conectar à API da DeepSeek. Tente novamente."
    if isinstance(erro, APIConnectionError):
        return "Não foi possível conectar à API da DeepSeek. Verifique sua conexão com a internet."
    return "Ocorreu um erro inesperado ao consultar a API da DeepSeek."


def _coagir_valor_do_campo(tipo: str, valor_recebido, valor_padrao_resolvido):
    """Garante que o valor devolvido pelo LLM para um campo seja compatível
    com o tipo esperado pelo modelo Pydantic. Se o LLM responder algo que não
    dá para validar (ex: texto num campo numérico), usamos o valor padrão
    daquele campo específico em vez de descartar a extração do abstract
    inteiro por causa de um único campo problemático."""
    if tipo == "número":
        try:
            return float(valor_recebido)
        except (TypeError, ValueError):
            return valor_padrao_resolvido
    if valor_recebido is None:
        return valor_padrao_resolvido
    return str(valor_recebido)


def extract_from_abstract(
    abstract: str,
    model_class,
    fields: list[dict],
    api_key: str,
    valor_padrao_global: str,
    modelo: str = MODELO_PADRAO,
) -> dict:
    """Extrai os campos definidos pelo aluno de um único abstract, usando a
    API da DeepSeek em modo JSON e validando a resposta com o modelo
    Pydantic gerado dinamicamente. Em caso de falha total (erro de rede,
    chave inválida, JSON ilegível, etc.), retorna todos os campos
    preenchidos com `valor_padrao_global` e uma chave extra "_erro" com uma
    mensagem amigável (removida pela interface antes de exibir a tabela)."""
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    system_prompt = build_system_prompt(fields, model_class)

    # Reserva tokens suficientes para o JSON de resposta, proporcional ao
    # número de campos, evitando que a resposta seja truncada no meio.
    max_tokens = max(512, 150 * len(fields))

    try:
        resposta = client.chat.completions.create(
            model=modelo,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": abstract},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=max_tokens,
            extra_body=THINKING_DESLIGADO,
        )
        dados_brutos = json.loads(resposta.choices[0].message.content)

        # Corrige campo a campo antes de validar, para que um valor
        # inesperado em UM campo não jogue fora os outros campos que o LLM
        # extraiu corretamente.
        dados_corrigidos = {}
        for campo in fields:
            valor_padrao_resolvido = resolve_default_value(campo["tipo"], campo["valor_padrao"])
            dados_corrigidos[campo["nome"]] = _coagir_valor_do_campo(
                campo["tipo"], dados_brutos.get(campo["nome"]), valor_padrao_resolvido
            )

        instancia = model_class.model_validate(dados_corrigidos)
        return instancia.model_dump()
    except Exception as erro:
        resultado_com_erro = {campo["nome"]: valor_padrao_global for campo in fields}
        resultado_com_erro["_erro"] = _mensagem_amigavel_para_erro(erro)
        return resultado_com_erro


def extract_batch(
    abstracts: list[str],
    model_class,
    fields: list[dict],
    api_key: str,
    valor_padrao_global: str,
    modelo: str = MODELO_PADRAO,
    progress_callback=None,
) -> list[dict]:
    """Processa uma lista de abstracts, chamando `extract_from_abstract` para
    cada um e notificando o progresso via `progress_callback(i, total)`."""
    resultados = []
    total = len(abstracts)
    for indice, abstract in enumerate(abstracts):
        resultado = extract_from_abstract(
            abstract, model_class, fields, api_key, valor_padrao_global, modelo
        )
        resultados.append(resultado)
        if progress_callback is not None:
            progress_callback(indice + 1, total)
    return resultados
