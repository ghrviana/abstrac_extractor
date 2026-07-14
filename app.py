"""
App Streamlit — Extração Estruturada de Dados de Abstracts Científicos.

Este é o arquivo principal: ele monta a interface (sidebar de configuração +
área principal com upload, execução e resultados) e orquestra as chamadas
para `model_builder.py` (gera o modelo Pydantic), `extractor.py` (chama a
DeepSeek) e `exporter.py` (gera Excel/CSV).
"""

import os
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from exporter import export_config_to_txt, export_to_csv, export_to_excel, results_to_dataframe
from extractor import build_system_prompt, extract_batch
from model_builder import build_extraction_model, generate_model_source, resolve_default_value, sanitize_field_name

# Carrega o arquivo .env (se existir) para preencher a chave de API
# automaticamente — útil em aula para não precisar redigitar a chave a cada
# vez que o app é reiniciado. O aluno também pode digitar a chave na hora,
# pela sidebar, sem precisar de nenhum arquivo .env.
load_dotenv()

st.set_page_config(page_title="Extração de Abstracts com LLM", page_icon="🧪", layout="wide")

# Valor usado quando a chamada à API falha por completo (erro de rede, chave
# inválida, etc.) — diferente do valor padrão de cada campo, que representa
# "essa informação não está no texto". Assim é possível distinguir os dois
# casos só olhando a tabela final.
VALOR_PADRAO_GLOBAL = "ERRO_NA_EXTRACAO"

MODELOS_DISPONIVEIS = ["deepseek-v4-flash", "deepseek-v4-pro"]

# --- Inicialização do estado da sessão --------------------------------------
if "api_key" not in st.session_state:
    st.session_state["api_key"] = os.getenv("DEEPSEEK_API_KEY", "")
if "modelo_deepseek" not in st.session_state:
    st.session_state["modelo_deepseek"] = MODELOS_DISPONIVEIS[0]
if "campos_ids" not in st.session_state:
    st.session_state["campos_ids"] = []
if "proximo_id_campo" not in st.session_state:
    st.session_state["proximo_id_campo"] = 0
if "resultado" not in st.session_state:
    st.session_state["resultado"] = None

st.title("🧪 Extração Estruturada de Dados de Abstracts Científicos")
st.caption(
    "Defina os campos que você quer extrair, carregue sua planilha de abstracts "
    "e gere uma tabela organizada usando um LLM — sem precisar programar."
)

# ============================================================================
# SEÇÃO 1 — Configuração da API (sidebar)
# ============================================================================
st.sidebar.header("🔑 Configuração da API")
st.session_state["api_key"] = st.sidebar.text_input(
    "Chave de API da DeepSeek",
    value=st.session_state["api_key"],
    type="password",
    help="Insira a chave de API fornecida pelo professor.",
)
st.session_state["modelo_deepseek"] = st.sidebar.selectbox(
    "Modelo DeepSeek",
    options=MODELOS_DISPONIVEIS,
    index=MODELOS_DISPONIVEIS.index(st.session_state["modelo_deepseek"]),
    help="Flash: mais rápido e econômico. Pro: maior capacidade de raciocínio, "
    "indicado para abstracts mais complexos.",
)

st.sidebar.divider()

# ============================================================================
# SEÇÃO 2 — Definição dos Campos de Extração (sidebar)
# ============================================================================
st.sidebar.header("📋 Campos para Extrair")

if st.sidebar.button("+ Adicionar Campo"):
    novo_id = st.session_state["proximo_id_campo"]
    st.session_state["campos_ids"].append(novo_id)
    st.session_state["proximo_id_campo"] += 1

campos_definidos = []
nomes_ja_usados = set()

for posicao, campo_id in enumerate(st.session_state["campos_ids"]):
    with st.sidebar.expander(f"Campo {posicao + 1}", expanded=True):
        nome_original = st.text_input(
            "Nome do campo", key=f"nome_{campo_id}", placeholder="ex: Proteína estudada"
        )
        tipo = st.selectbox("Tipo de dado", ["texto", "número", "lista"], key=f"tipo_{campo_id}")
        descricao = st.text_area(
            "Descrição/dica para o modelo",
            key=f"descricao_{campo_id}",
            placeholder="ex: Nome da proteína principal estudada no abstract",
        )
        valor_padrao = st.text_input(
            "Valor quando não encontrado", value="Não informado", key=f"padrao_{campo_id}"
        )

        if st.button("🗑️ Remover", key=f"remover_{campo_id}"):
            st.session_state["campos_ids"].remove(campo_id)
            st.rerun()

        if nome_original.strip():
            nome_sanitizado = sanitize_field_name(nome_original)
            # Evita que dois campos com nomes parecidos (ex: "Espécie" e
            # "espécie ") gerem a mesma chave e se sobrescrevam silenciosamente.
            nome_final = nome_sanitizado
            sufixo = 2
            while nome_final in nomes_ja_usados:
                nome_final = f"{nome_sanitizado}_{sufixo}"
                sufixo += 1
            nomes_ja_usados.add(nome_final)

            campos_definidos.append(
                {
                    "nome": nome_final,
                    "label": nome_original.strip(),
                    "tipo": tipo,
                    "descricao": descricao,
                    "valor_padrao": valor_padrao,
                }
            )

if not campos_definidos:
    st.sidebar.warning("Nenhum campo definido ainda. Clique em '+ Adicionar Campo' para começar.")

# ============================================================================
# SEÇÃO 3 — Upload e Configuração do Arquivo (área principal)
# ============================================================================
st.header("1️⃣ Carregue sua planilha de abstracts")

arquivo = st.file_uploader("Selecione um arquivo .xlsx ou .csv", type=["xlsx", "csv"])

df_abstracts = None
coluna_abstract = None
colunas_chave = []
linhas_selecionadas = None

if arquivo is not None:
    try:
        if arquivo.name.lower().endswith(".csv"):
            df_abstracts = pd.read_csv(arquivo)
        else:
            df_abstracts = pd.read_excel(arquivo)
    except Exception:
        st.error("❌ Não foi possível ler o arquivo. Verifique se ele está no formato correto (.xlsx ou .csv).")
        df_abstracts = None

    if df_abstracts is not None:
        total_linhas = len(df_abstracts)

        st.write(f"Prévia das 5 primeiras linhas (total de **{total_linhas}** linhas na planilha):")
        st.dataframe(df_abstracts.head(5), use_container_width=True)

        coluna_abstract = st.selectbox(
            "Selecione a coluna que contém os abstracts", options=df_abstracts.columns.tolist()
        )
        colunas_chave = st.multiselect(
            "Colunas para usar como chave de merge (opcional)",
            options=[c for c in df_abstracts.columns.tolist() if c != coluna_abstract],
            max_selections=2,
            help="Selecione até 2 colunas da planilha original (ex: ID, DOI) para incluí-las "
            "sem alteração na tabela final, permitindo unir o resultado de volta com a "
            "planilha de origem.",
        )

        st.markdown("**Quantos abstracts você quer processar?**")
        modo_selecao = st.radio(
            "Quantos abstracts você quer processar?",
            options=["Todos os abstracts", "Número máximo de abstracts", "Intervalo de linhas"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if modo_selecao == "Todos os abstracts":
            st.info(f"ℹ️ Serão processados todos os **{total_linhas}** abstracts da planilha.")
            linhas_selecionadas = df_abstracts

        elif modo_selecao == "Número máximo de abstracts":
            limite_abstracts = st.number_input(
                "Número máximo de abstracts a processar",
                min_value=1,
                max_value=total_linhas,
                value=min(10, total_linhas),
            )
            linhas_selecionadas = df_abstracts.head(int(limite_abstracts))

        else:  # Intervalo de linhas
            if total_linhas < 2:
                st.info("A planilha tem apenas 1 linha de dados — ela será processada.")
                linhas_selecionadas = df_abstracts
            else:
                linha_inicial, linha_final = st.slider(
                    "Intervalo de linhas a processar",
                    min_value=1,
                    max_value=total_linhas,
                    value=(1, min(10, total_linhas)),
                    help="Linha 1 = primeira linha de dados da planilha (o cabeçalho não conta).",
                )
                linhas_selecionadas = df_abstracts.iloc[linha_inicial - 1 : linha_final]
                st.caption(
                    f"Serão processados **{len(linhas_selecionadas)}** abstracts "
                    f"(linha {linha_inicial} a {linha_final})."
                )

# ============================================================================
# SEÇÃO 4 — Execução da Extração
# ============================================================================
st.header("2️⃣ Execute a extração")

botao_desabilitado = (
    not st.session_state["api_key"]
    or len(campos_definidos) == 0
    or arquivo is None
    or df_abstracts is None
)

if st.button("🚀 Iniciar Extração", disabled=botao_desabilitado, type="primary"):
    rotulos_dos_campos = {campo["label"] for campo in campos_definidos}
    colunas_em_conflito = set(colunas_chave) & (rotulos_dos_campos | {"abstract_original"})
    if colunas_em_conflito:
        st.error(
            "❌ As colunas de chave selecionadas têm nomes que colidem com colunas "
            f"geradas pelo app: {', '.join(sorted(colunas_em_conflito))}. Renomeie a coluna "
            "na planilha original ou o campo extraído e tente novamente."
        )
        st.stop()

    modelo_extracao = build_extraction_model(campos_definidos)
    abstracts_selecionados = linhas_selecionadas[coluna_abstract].astype(str).tolist()
    df_colunas_chave = linhas_selecionadas[colunas_chave] if colunas_chave else None

    barra_progresso = st.progress(0.0)
    with st.status("Processando abstracts...", expanded=True) as status:

        def atualizar_progresso(atual, total):
            barra_progresso.progress(atual / total)
            status.update(label=f"Processando abstract {atual} de {total}...")

        resultados = extract_batch(
            abstracts_selecionados,
            modelo_extracao,
            campos_definidos,
            st.session_state["api_key"],
            VALOR_PADRAO_GLOBAL,
            modelo=st.session_state["modelo_deepseek"],
            progress_callback=atualizar_progresso,
        )
        status.update(label="Extração concluída!", state="complete")

    # Mensagens de erro (se houver) ficam numa chave especial "_erro" que não
    # deve aparecer na tabela final — por isso são removidas aqui com pop().
    mensagens_de_erro = sorted({r.pop("_erro") for r in resultados if "_erro" in r})

    df_resultado = results_to_dataframe(resultados, abstracts_selecionados, df_colunas_chave)
    mapa_labels = {campo["nome"]: campo["label"] for campo in campos_definidos}
    df_resultado = df_resultado.rename(columns=mapa_labels)

    st.session_state["resultado"] = {
        "df": df_resultado,
        "campos": campos_definidos,
        "modelo_fonte": generate_model_source(campos_definidos),
        "system_prompt_exemplo": build_system_prompt(campos_definidos, modelo_extracao),
        # Carimbado uma única vez aqui (não na hora de montar os botões de
        # download) porque o Streamlit reexecuta o script inteiro a cada
        # interação — se calculássemos o timestamp na seção de download, o
        # nome do arquivo mudaria a cada rerender, mesmo sem nova extração.
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    }

    if mensagens_de_erro:
        texto_erros = "\n".join(f"- {mensagem}" for mensagem in mensagens_de_erro)
        st.error(f"⚠️ Ocorreram erros durante a extração de alguns abstracts:\n{texto_erros}")

    st.success(f"✅ {len(resultados)} abstracts processados com sucesso!")

# ============================================================================
# SEÇÃO 5 — Prévia e Download dos Resultados
# ============================================================================
if st.session_state["resultado"] is not None:
    st.header("3️⃣ Resultados")

    df_resultado = st.session_state["resultado"]["df"]
    campos_usados = st.session_state["resultado"]["campos"]

    st.subheader("📊 Prévia dos Dados Extraídos")
    st.dataframe(df_resultado, use_container_width=True)

    total_abstracts = len(df_resultado)
    total_campos = len(campos_usados)
    total_celulas = total_abstracts * total_campos

    celulas_com_valor_padrao = 0
    for campo in campos_usados:
        valor_padrao_resolvido = resolve_default_value(campo["tipo"], campo["valor_padrao"])
        valores_considerados_padrao = {valor_padrao_resolvido, VALOR_PADRAO_GLOBAL}
        celulas_com_valor_padrao += df_resultado[campo["label"]].isin(valores_considerados_padrao).sum()

    percentual_padrao = (celulas_com_valor_padrao / total_celulas * 100) if total_celulas else 0.0

    coluna_a, coluna_b, coluna_c = st.columns(3)
    coluna_a.metric("Abstracts processados", total_abstracts)
    coluna_b.metric("Campos extraídos por abstract", total_campos)
    coluna_c.metric("% células com valor padrão", f"{percentual_padrao:.1f}%")

    excel_bytes = export_to_excel(
        df_resultado,
        st.session_state["resultado"]["modelo_fonte"],
        st.session_state["resultado"]["system_prompt_exemplo"],
    )
    csv_bytes = export_to_csv(df_resultado)
    config_txt_bytes = export_config_to_txt(
        st.session_state["resultado"]["modelo_fonte"],
        st.session_state["resultado"]["system_prompt_exemplo"],
    )
    timestamp = st.session_state["resultado"]["timestamp"]

    coluna_download_1, coluna_download_2, coluna_download_3 = st.columns(3)
    coluna_download_1.download_button(
        "⬇️ Baixar Excel (.xlsx)",
        data=excel_bytes,
        file_name=f"extracao_abstracts_{timestamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    coluna_download_2.download_button(
        "⬇️ Baixar CSV (.csv)",
        data=csv_bytes,
        file_name=f"extracao_abstracts_{timestamp}.csv",
        mime="text/csv",
    )
    coluna_download_3.download_button(
        "⬇️ Baixar Configuração (.txt)",
        data=config_txt_bytes,
        file_name=f"configuracao_extracao_{timestamp}.txt",
        mime="text/plain",
    )

# ============================================================================
# SEÇÃO 6 — Painel Educacional
# ============================================================================
with st.expander("🎓 O que está acontecendo nos bastidores?"):
    st.markdown(
        """
1. **Upload**: sua planilha é lida com `pandas` e você escolhe qual coluna contém o texto dos abstracts.
2. **Modelo Pydantic dinâmico**: para cada campo que você define, o app gera automaticamente uma classe Python (com `pydantic.create_model()`) que descreve o formato exato dos dados esperados — nome, tipo e valor padrão de cada campo.
3. **Chamada ao LLM**: o app envia o abstract e um *system prompt* para a API da DeepSeek, pedindo que a resposta venha em JSON, seguindo o formato do modelo.
4. **Parse estruturado**: a resposta do modelo é validada com a classe Pydantic gerada — se algo vier fora do esperado, o valor padrão é usado no lugar, sem quebrar o processamento dos outros abstracts.
5. **Exportação**: os resultados de todos os abstracts são organizados em uma única tabela e exportados em Excel ou CSV.
        """
    )

    if st.session_state["resultado"] is not None:
        st.markdown("**Modelo Pydantic gerado para a última extração:**")
        st.code(st.session_state["resultado"]["modelo_fonte"], language="python")

        st.markdown("**System prompt enviado à DeepSeek na última extração:**")
        st.code(st.session_state["resultado"]["system_prompt_exemplo"], language="text")
    else:
        st.info("Execute uma extração para ver aqui o modelo Pydantic e o prompt gerados dinamicamente.")
