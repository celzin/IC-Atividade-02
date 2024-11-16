<div align="center" style="display: inline_block">
  <img align="center" alt="VS" src="https://img.shields.io/badge/Visual_Studio_Code-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white" />
  <img align="center" alt="Linux" src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black" />
  <img align="center" alt="Python" src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
</div>

# 🗂️​ Estrutura do Repositório

- `src/`: Diretório com os scripts Python usados para gerar os gráficos e realizar as análises.
- `docs/`: Diretório contendo as instruções do trabalho e o Relatório de análise em PDF.
<!-- - `imgs/`: Diretório com as imagens utilizadas na documentação. -->

# 📝 Resumo

<div align="justify">
<!-- Este repositório apresenta o desenvolvimento e a análise de diversas operações e composições fuzzy aplicadas ao estudo de relações fuzzy. As implementações exploradas incluem as operações fuzzy (<code>Complemento</code>, <code>União</code>, <code>Interseção</code>, <code>T-Normas</code>, <code>S-Normas</code>) e das composições (<code>Max-Min</code>, <code>Min-Max</code> e <code>Max-Prod</code>). Esses métodos são avaliados com base em seu impacto nas relações fuzzy entre variáveis do universo em análise, permitindo uma análise detalhada das interações entre conjuntos fuzzy. O cenário de aplicação envolve conjuntos de altura e idade, com o objetivo de investigar a eficácia de diferentes composições e operações fuzzy. Os resultados destacam as variações que cada método proporciona na representação das relações fuzzy, possibilitando uma análise comparativa das abordagens em termos de flexibilidade e precisão. -->
</div>

# 🔄 Compilação e Execução 

<div align="justify">
Para executar o programa, siga os passos abaixo:

- Abra o terminal no diretório onde os arquivos do projeto estão localizados.
- Certifique-se de que as bibliotecas `networkx` e `matplotlib` estão instaladas. Se não estiverem instaladas, você pode instalar essas bibliotecas manualmente utilizando o seguinte comando:

```bash
pip install numpy matplotlib seaborn
```

- Em seguida, execute o programa com o comando:

```bash
python main.py
```

</div>

# 📞 Contato

<table align="center">
  <tr>
    <th>Contribuinte</th>
    <th>Contato</th>
  </tr>
  <tr>
    <td>Celso</td>
    <td><a href="https://t.me/celso_vsf"><img align="center" height="20px" width="90px" src="https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white"/> </td>
  </tr>
</table>

<!-- Inteligência Computacional

Atividade - Sistema Fuzzy Funcional

Implementar um sistema fuzzy de Takagi-Sugeno de ordem zero ou primeira ordem para aproximar a seguinte função não linear: f(x) = e^(−x/5) · sin(3x) + 0.5 · sin(x). A aproximação deve estar no intervalo x ∈[0, 10]. O sistema deve aproximar essa função com o menor erro possível dentro do intervalo especificado.

Etapas:

1. Ideia Geral
  • Gere um conjunto de dados com valores de x no intervalo de [0, 10] e calcule os valores correspondentes de f(x) para esses pontos. Atenção para gerar um número suficiente de pontos.
  • Defina a estrutura do sistema fuzzy de Takagi-Sugeno (ordem zero ou primeira ordem).
  • Escolha as variáveis linguísticas e defina funções de pertinência que cubram o intervalo de x.
  • Teste diferentes combinações de operadores fuzzy e funções de pertinência. Lembrem-se, inicialmente fazer usando o padrão (benckmark) e depois fazer as variações.
  • Avalie o desempenho do modelo comparando os valores aproximados pelo sistema com os valores reais de f(x) no intervalo de [0, 10]. Use o MSE (Mean Square Error) ou RMSE (Root Mean Square Error) para a comparação. Otimizar a solução tendo em vista reduzir o erro de aproximação.
  • Se necessário, pode usar o RLS (Recursive Least Square) ou Gradiente Descendente para encontrar os valores dos parâmetros do consequente.
  • O sistema deve ser desenvolvido em linha de código e pode ser implementado em Python. Não devem ser utilizadas ToolBoxs, Bibliotecas, etc. Porém, para implementar o RLS, Gradiente e Visualização Gráfica podem ser utilizadas estruturas de código prontas e bibliotecas.

2. Apresente um relatório contendo (Saída no terminal):
  • Descrição detalhada do sistema fuzzy desenvolvido, incluindo variáveis linguísticas, funções de pertinência, operadores fuzzy e funções de pertinência utilizadas.
  • Gráficos comparando a curva da função f(x) com a curva aproximada pelo seu modelo fuzzy.
  • Gráficos ilustrando o erro.
  • A métrica de erro final obtida e uma análise sobre as combinações testadas, destacando as configurações que trouxeram melhores resultados.

3. Critérios de Avaliação
  • Configuração correta e detalhada do sistema fuzzy de Takagi-Sugeno.
  • Qualidade e precisão da aproximação da função alvo f(x).
  • Variedade e análise das combinações testadas (operadores, funções de ativação, etc.).
  • Clareza na apresentação do relatório e dos resultados obtidos.

4. Dicas
  • Explore diferentes tipos de funções de pertinência e operadores fuzzy. Testar várias combinações pode revelar configurações que melhor se adaptam ao comportamento complexo da função.
  • Analise o impacto das mudanc ̧as nos operadores e nas funções de ativação para otimizar a modelagem.
  • As etapas listadas são somente um direcionamento do que deve ser feito. -->