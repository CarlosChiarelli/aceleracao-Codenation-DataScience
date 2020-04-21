import streamlit as st
import pandas as pd
# para download de arquivo
import base64

# gera um link para download de um arquivo
def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return href

def main():
    st.image('auctus_logo.png', width=400)
    st.title('AUCTUS.ai - Inteligência Aumentada')
    st.audio('audio.mp4')
    st.header('Aplicação StreamLit gerada em Python')
    st.subheader('Explorando funcionalidades')
    st.text('Podemos fazer importação, manipulação, visualização interativa e modelagem de dados')

    # criando botão
    botao = st.button('Botão')
    # se cliquei no botao:
    if botao:
        st.markdown('Clicado')

    # criando checkbox
    check = st.checkbox('Checkbox')
    if check:
        st.markdown('Checkbox clicado')

    # lista de opções
    st.markdown('Lista de opções')
    listaOpcoes = st.radio('escolha as opções', ('Op1', 'Op2'))
    if listaOpcoes == 'Op1':
        st.markdown('opcao 01')
    if listaOpcoes == 'Op2':
        st.markdown('opcao 02')

    # caixa de seleção
    selecao = st.selectbox('lista das opções', ('Op1', 'Op2'))
    if selecao == 'Op1':
        st.markdown('opcao 01 (padrão)')
    if selecao == 'Op2':
        st.markdown('opcao 02 !!')

    # multi selecao
    st.markdown('multiseleção')
    multi = st.multiselect('escolha cor:', ('Op1', 'Op2'))
    if selecao == 'Op1':
        st.markdown('azul')
    if selecao == 'Op2':
        st.markdown('vermelho')

    # importando arquivo file_uploader
    st.markdown('Carregamento de arquivo')
    # vai aceitas APENAS csv
    arquivo = st.file_uploader('escolha seu arquivo', type='csv')
    # se eu já subi algum arquivo (não está vazio)
    if arquivo is not None:
        st.markdown('arquivo não está vazio !!!')

    # importando .csv da máquina local
    #df = pd.read_csv('IRIS.csv')

    arquivo=1
    if arquivo is not None:
        #df = pd.read_csv(arquivo)
        df = pd.read_csv('IRIS.csv')
        # visualização do df

        # criando botão (range) dinamico de 1 a 100
        slider = st.slider('valores', 1, 100)
        st.dataframe(df.head(slider))

        # tabela em Markdown
        st.markdown('tabela em MARKDOWN')
        st.table(df.head(slider))

        # escrevendo info na tela
        st.markdown('nomes das colunas')
        st.write(df.columns)

        # sumarizando dados (média dos grupos)
        st.markdown('Média do comprimento das pétalas das espécies')
        st.table( df.groupby('species')['petal_width'].mean() )


    st.markdown('\n\n\n ____ \n\n\n')
    # código pré-pronto
    #file  = st.file_uploader('Escolha a base de dados que deseja analisar (.csv)', type = 'csv')
    file = 'IRIS.csv'
    if file is not None:
        st.subheader('Analisando os dados')
        df = pd.read_csv(file)

        st.markdown('**Número de linhas:**')
        st.markdown(df.shape[0])

        st.markdown('**Número de colunas:**')
        st.markdown(df.shape[1])

        st.markdown('**Visualizando o dataframe**')
        number = st.slider('Escolha o numero de colunas que deseja ver', min_value=1, max_value=20)
        st.dataframe(df.head(number))

        st.markdown('**Nome das colunas:**')
        st.markdown(list(df.columns))

        # nomes colunas, tipos colunas, NA total e percentual
        exploracao = pd.DataFrame({'nomes' : df.columns, 'tipos' : df.dtypes, 'NA #': df.isna().sum(), 'NA %' : (df.isna().sum() / df.shape[0]) * 100})

        st.markdown('**Contagem dos tipos de dados:**')
        st.write(exploracao.tipos.value_counts())

        st.markdown('**Nomes das colunas do tipo int64:**')
        st.markdown(list(exploracao[exploracao['tipos'] == 'int64']['nomes']))
        st.markdown('**Nomes das colunas do tipo float64:**')
        st.markdown(list(exploracao[exploracao['tipos'] == 'float64']['nomes']))
        st.markdown('**Nomes das colunas do tipo object:**')
        st.markdown(list(exploracao[exploracao['tipos'] == 'object']['nomes']))

        st.markdown('**Tabela com coluna e percentual de dados faltantes :**')
        st.table(exploracao[exploracao['NA #'] != 0][['tipos', 'NA %']])

        st.subheader('Inputaçao de dados númericos :')
        percentual = st.slider('Escolha o limite de percentual faltante limite para as colunas vocë deseja inputar os dados', min_value=0, max_value=100)
        # colunas com percentual escolhido
        lista_colunas = list(exploracao[exploracao['NA %']  < percentual]['nomes'])
        # forma de inputação
        select_method = st.radio('Escolha um metodo abaixo :', ('Média', 'Mediana'))
        st.markdown('Você selecionou : ' +str(select_method))
        # imputando os dados com a opção escolhida
        # média
        if select_method == 'Média':
            df_inputado = df[lista_colunas].fillna(df[lista_colunas].mean())
            exploracao_inputado = pd.DataFrame({'nomes': df_inputado.columns,
                                                'tipos': df_inputado.dtypes,
                                                'NA #': df_inputado.isna().sum(),
                                                'NA %': (df_inputado.isna().sum() / df_inputado.shape[0]) * 100})
            st.table(exploracao_inputado[exploracao_inputado['tipos'] != 'object']['NA %'])
            st.subheader('Dados Inputados faça download abaixo : ')
            st.markdown(get_table_download_link(df_inputado), unsafe_allow_html=True)
        # mediana
        if select_method == 'Mediana':
            df_inputado = df[lista_colunas].fillna(df[lista_colunas].mean())
            exploracao_inputado = pd.DataFrame({'nomes': df_inputado.columns,
                                                'tipos': df_inputado.dtypes,
                                                'NA #': df_inputado.isna().sum(),
                                                'NA %': (df_inputado.isna().sum() / df_inputado.shape[0]) * 100})
            st.table(exploracao_inputado[exploracao_inputado['tipos'] != 'object']['NA %'])
            st.subheader('Dados Inputados faça download abaixo : ')
            st.markdown(get_table_download_link(df_inputado), unsafe_allow_html=True)


if __name__ == '__main__':
	main()
