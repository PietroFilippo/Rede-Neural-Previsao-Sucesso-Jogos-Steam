import numpy as np
import matplotlib.pyplot as plt

# Dados para o treinamento
preco = np.array([0.00, 0.00, 274.00, 199.90, 249.90, 90.99, 299.90, 229.90, 69.00, 199.00, 0.00, 299.90, 199.00, 249.00]) # preço do jogo convertido em R$, 0 = free to play (gratuito para jogar)
review_geral = np.array([4.35, 4.10, 3.25, 4.05, 2.10, 4.30, 3.55, 4.60, 1.25, 2.60, 4.65, 0.60, 2.60, 4.85]) # reviews gerais do jogo, 5.0 = máxima, 0.0 = mínimo
multiplayer_ou_singleplayer = np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0])  # 1 para multiplayer, 0 para singleplayer

# Saída desejada (indicando o sucesso)
successo = np.array([1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1]) # 1 para sucesso, 0 para "fracasso"

# Número de épocas e padrões
numEpocas = 250000
q = 14

# Taxa de aprendizado
eta = 0.01

# Inicialização aleatória das matrizes de pesos
N = 2  # Número de neurônios na primeira camada oculta
L = 1  # Número de neurônios na camada de saída
M = 3  # Número de recursos (preco, review_geral, multiplayer_ou_singleplayer)

M2 = 4  # Número de neurônios na segunda camada oculta
M3 = 3  # Número de neurônios na terceira camada oculta
eta2 = 0.01  # Taxa de aprendizado para a camada oculta

W1 = np.random.random((N, M + 1))
W2 = np.random.random((M2, N + 1))
W3 = np.random.random((M3, M2 + 1))
W4 = np.random.random((L, M3 + 1))

# Array para armazenar os erros
E = np.zeros(q)
Etm = np.zeros(numEpocas)

# Bias
bias = 1

# Entrada do Perceptron
X = np.vstack((preco, review_geral, multiplayer_ou_singleplayer))

# TREINAMENTO
for i in range(numEpocas):
    for j in range(q):
        Xb = np.hstack((bias, X[:, j]))

        o1 = np.tanh(W1.dot(Xb))
        o1b = np.insert(o1, 0, bias)

        o2 = np.tanh(W2.dot(np.insert(o1, 0, bias)))
        o2b = np.insert(o2, 0, bias)
        
        o3 = np.tanh(W3.dot(np.insert(o2, 0, bias)))
        o3b = np.insert(o3, 0, bias)

        Y = np.tanh(W4.dot(o3b))
        e = successo[j] - Y

        E[j] = (e.transpose().dot(e)) / 2

        delta4 = np.diag(e).dot((1 - Y * Y))
        vdelta4 = (W4.transpose()).dot(delta4)
        delta3 = np.diag(1 - o3b * o3b).dot(vdelta4)
        vdelta3 = W3.transpose().dot(delta3[:-1])
        delta2 = np.diag(1 - o2b * o2b).dot(vdelta3)
        vdelta2 = W2.transpose().dot(delta2[:-1]) 
        delta1 = np.diag(1 - o1b * o1b).dot(vdelta2)

        W1 = W1 + eta * (np.outer(delta1[1:], Xb))
        W2 = W2 + eta * (np.outer(delta2[1:], o1b))
        W3 = W3 + eta * (np.outer(delta3[1:], o2b))
        W4 = W4 + eta2 * (np.outer(delta4, o3b))

    Etm[i] = E.mean()

# Plotagem do erro médio
plt.xlabel("Épocas")
plt.ylabel("Erro Médio")
plt.plot(Etm, color='b')
plt.plot(Etm)
plt.show()

# TESTE DA REDE
# Exemplo de um novo jogo com informações do preço, avaliação geral e se é singleplayer ou multiplayer
novo_jogo = np.array([0.00, 5.00, 0])  # Dados do novo jogo

# Inserir o bias na entrada do novo jogo
novo_jogo_com_bias = np.hstack((bias, novo_jogo))

# Calcular a saída da primeira camada oculta
output1 = np.tanh(W1.dot(novo_jogo_com_bias))
output1_com_bias = np.insert(output1, 0, bias)

# Calcular a saída da segunda camada oculta
output2 = np.tanh(W2.dot(output1_com_bias))
output2_com_bias = np.insert(output2, 0, bias)

# Calcular a saída da terceira camada oculta
output3 = np.tanh(W3.dot(output2_com_bias))
output3_com_bias = np.insert(output3, 0, bias)

# Calcular a saída da rede neural
output_previsto = np.tanh(W4.dot(output3_com_bias))

# Imprimir a previsão da rede para o novo jogo
print("Previsão da rede para o novo jogo:", output_previsto)

