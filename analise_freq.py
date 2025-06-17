import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os # Para verificar a existência do arquivo

def analisar_audio_fft(caminho_arquivo_audio):
    """
    Carrega um arquivo de áudio, calcula sua FFT e exibe o espectro de frequência.

    Args:
        caminho_arquivo_audio (str): O caminho completo para o arquivo de áudio (e.g., .wav, .mp3).
    """

    # 1. Verificar se o arquivo existe
    if not os.path.exists(caminho_arquivo_audio):
        print(f"Erro: O arquivo '{caminho_arquivo_audio}' não foi encontrado.")
        print("Por favor, verifique o caminho e o nome do arquivo.")
        return

    # 2. Carregar o arquivo de áudio
    print(f"Carregando o arquivo de áudio: {caminho_arquivo_audio}...")
    try:
        y, sr = librosa.load(caminho_arquivo_audio)
        # y: série temporal do áudio (amostras)
        # sr: taxa de amostragem (sample rate) do áudio
        print(f"Áudio carregado com sucesso. Taxa de amostragem (SR): {sr} Hz.")
    except Exception as e:
        print(f"Erro ao carregar o arquivo de áudio: {e}")
        print("Certifique-se de que o arquivo é um formato de áudio suportado (.wav, .mp3, etc.)")
        return

    # 3. Calcular a Transformada Rápida de Fourier (FFT)
    # A FFT de um sinal real é simétrica, então pegamos apenas a primeira metade.
    # np.fft.fft retorna um array de números complexos.
    # np.abs calcula a magnitude (amplitude) desses números complexos.
    print("Calculando a FFT...")
    N = len(y) # Número total de amostras
    yf = np.fft.fft(y)
    xf = np.fft.fftfreq(N, 1 / sr) # Frequências correspondentes às amostras da FFT

    # Como o sinal de áudio é real, o espectro da FFT é simétrico.
    # Nos interessam apenas as frequências positivas (primeira metade do espectro).
    # O +1 é para incluir a frequência Nyquist se N for par
    primeira_metade_indices = np.arange(1, N // 2 + 1)
    yf_primeira_metade = yf[primeira_metade_indices]
    xf_primeira_metade = xf[primeira_metade_indices]

    # Calcular a magnitude (amplitude) do espectro de frequência
    # Usamos 2/N para normalizar a amplitude para que corresponda à amplitude original do sinal
    # (considerando que pegamos apenas metade do espectro)
    magnitude_espectro = 2.0/N * np.abs(yf_primeira_metade)

    # 4. Visualizar o espectro de frequência
    print("Gerando o gráfico do espectro de frequência...")
    plt.figure(figsize=(15, 6))

    # Plotar o espectro de frequência
    plt.plot(xf_primeira_metade, magnitude_espectro)
    plt.title(f'Espectro de Frequência do Áudio: {os.path.basename(caminho_arquivo_audio)}')
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Magnitude (Amplitude Normalizada)')
    plt.grid(True)
    plt.xscale('log') # Escala logarítmica para as frequências para melhor visualização
    plt.xlim([20, sr/2]) # Limita o eixo X para a faixa audível (20 Hz a Nyquist)

    plt.tight_layout()
    plt.show()

    print("Análise de FFT concluída.")

# --- Como usar o código ---
if __name__ == "__main__":
    # COLOQUE O CAMINHO PARA A SUA MÚSICA AQUI
    # Exemplo: Se sua música estiver na mesma pasta que o script, use apenas o nome do arquivo.
    # Se estiver em outra pasta, use o caminho completo (e.g., 'C:/Musicas/minha_musica.mp3' no Windows, ou '/home/usuario/musicas/minha_musica.wav' no Linux/macOS)
    caminho_da_sua_musica = 'Dancing Alone.mp3' # <--- ALtere este caminho para o seu arquivo de áudio!

    # Para testar, você pode baixar um pequeno arquivo .wav ou .mp3 de exemplo.
    # Certifique-se de que o arquivo de áudio esteja no mesmo diretório do seu script Python,
    # ou forneça o caminho completo para ele.

    analisar_audio_fft(caminho_da_sua_musica)