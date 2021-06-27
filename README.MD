# Análise de dataset e deploy com Streamlit

Neste projeto iremos analisar alguns datasets famosos que são built in do pacote datasets
A ideia principal é mostrar o poder da ferramenta Streamlit, que é uma boa alternativa para o Flask e Django.
Neste exemplo vamos usar a IDE PyCharm

## Instalação

* 1 - pip install straemlit
* 2 - pip install scikit-learn
* 3 - pip install matplotlib
* 4 - pip install numpy
* 5 - pip install pandas
* 6 - pip install sklearn

Para testar a instalação do streamlit você pode abrir no seu terminal o seguinte comando:
streamlit hello
Após isso, você será redirecionado para uma página local, provavelmente no endereço: http://localhost:8501
Aqui você tem algumas demonstrações do Streamlit

Após a demonstração, começaremos a criar nosso projeto..
Volte ao terminar e clique em Ctrl + C para paralisar a atual instância do Streamlit

Crie um arquivo com a extensão .py
Exemplo: main.py
Vomos importar o pacote do Streamlit
import streamlit as st
Vamos criar um título para o nosso projeto
'''
st.title("Análise de Datasets com Streamlit")
'''

Agora você pode executar o streamlit novamente para ver como está o resultado
Para isso basta executar o arquivo .py no terminal:
streamlit run main.py
Quando abrir no browser, você pode habilitar a função para que a página seja atualizada em modo de desenvolvimento.
Para isso, vá ao menu direito, clique em Settings >> Development >> selecione "Run on Save"
Desta forma tudo será atualizado quando seu código for atualizado

Para escrevermos no corpo do nosso projeto, basta usar a função st.write, vamos usar as marcações:
st.write('''
# Explorando diferentes classificadores
Qual é o melhor?
''')

Também é possível criar selectbox lateral, para isso vamos escolher alguns dos datasets mais populares:
dataset_name = st.sidebar.selectbox("Selecione o Dataset", ("Iris", "Breast Cancer", "Wine Dataset"))
st.write(dataset_name)

Agora crie um selectbox para os classificadores:
classifier_name = st.sidebar.selectbox("Selecione o Classificador", ("KNN", "SVM", "Random Forest"))

Crie a função que será responsável por carregar os datasets de acordo com a escolha do usuário:
```
def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
```

Vamos criar outra função para carregar os parâmetros de acordo com o modelo selecionado:
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

Vamos criar uma função para carregar os modelos e seus respectivos parametros anteriormente definidos:
def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(max_depth=params["max_depth"],
                                     n_estimators=params["n_estimators"], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)

Agora vamos realizar a classificação dos modelos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1234)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"Classificador = {classifier_name}")
st.write(f"Acurácia = {acc}")

Para realizarmos a plotagem é necessário fazermos a redução de dimensionalidade com PCA
#Plot
pca = PCA(2)
X_projected = pca.fit_transform(X)
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
st.pyplot(fig)








```bash
pip install foobar
```

## Usage

```python
import foobar

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
