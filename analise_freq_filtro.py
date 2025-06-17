import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, lfilter
import soundfile as sf 

def butter_bandstop(lowcut, highcut, fs, order=5):
    """
    Define os coeficientes de um filtro Butterworth passa-banda.
    lowcut, highcut: frequências de corte (Hz)
    fs: taxa de amostragem (Hz)
    order: ordem do filtro
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='bandstop')
    return b, a

def apply_bandstop_filter(data, lowcut, highcut, fs, order=5):
    """
    Aplica o filtro passa-banda ao sinal de áudio.
    """
    b, a = butter_bandstop(lowcut, highcut, fs, order=order)
    y_filtered = lfilter(b, a, data)
    return y_filtered

def analisar_e_filtrar_audio(caminho_arquivo_audio, nome_arquivo_saida='audio_instrumental.wav'):
    """
    Carrega um arquivo de áudio, calcula FFTs (original e filtrada),
    aplica um filtro para tentar remover vocais e salva o resultado.
    """

    if not os.path.exists(caminho_arquivo_audio):
        print(f"Erro: O arquivo '{caminho_arquivo_audio}' não foi encontrado.")
        print("Por favor, verifique o caminho e o nome do arquivo.")
        return

    print(f"Carregando o arquivo de áudio: {caminho_arquivo_audio}...")
    try:
        # sr=None para manter o SR original do arquivo de áudio
        y_original, sr = librosa.load(caminho_arquivo_audio, sr=None)
        print(f"Áudio carregado com sucesso. Taxa de amostragem (SR): {sr} Hz.")
    except Exception as e:
        print(f"Erro ao carregar o arquivo de áudio: {e}")
        print("Certifique-se de que o arquivo é um formato de áudio suportado (.wav, .mp3, etc.)")
        return

    # --- 1. FFT e Plot do Áudio Original ---
    print("Calculando e plotando FFT do áudio original...")
    N_original = len(y_original)
    yf_original = np.fft.fft(y_original)
    xf_original = np.fft.fftfreq(N_original, 1 / sr)

    # Pegar apenas a primeira metade das frequências (positivas)
    primeira_metade_indices_original = np.arange(1, N_original // 2 + 1)
    xf_primeira_metade_original = xf_original[primeira_metade_indices_original]
    magnitude_espectro_original = 2.0/N_original * np.abs(yf_original[primeira_metade_indices_original])

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1) # 2 linhas, 1 coluna, primeiro gráfico
    plt.plot(xf_primeira_metade_original, magnitude_espectro_original)
    plt.title(f'Espectro de Frequência Original')
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Magnitude (Amplitude Normalizada)')
    plt.grid(True)
    plt.xscale('log')
    plt.xlim([20, sr/2]) # Faixa audível

    # --- 2. Aplicação do Filtro para Atenuar Voz ---
    print("Aplicando filtro para atenuar frequências vocais...")
    
    lowcut_vocal = 300  # Hz (frequência de corte inferior para a banda de rejeição)
    highcut_vocal = 5000 # Hz (frequência de corte superior para a banda de rejeição)
    filter_order = 8 # Ordem do filtro (quanto maior, mais "íngreme" o corte, mas mais artefatos)

    y_filtered = apply_bandstop_filter(y_original, lowcut_vocal, highcut_vocal, sr, order=filter_order)

    # --- 3. FFT e Plot do Áudio Filtrado ---
    print("Calculando e plotando FFT do áudio filtrado...")
    N_filtered = len(y_filtered)
    yf_filtered = np.fft.fft(y_filtered)
    xf_filtered = np.fft.fftfreq(N_filtered, 1 / sr)

    # Pegar apenas a primeira metade das frequências (positivas)
    primeira_metade_indices_filtered = np.arange(1, N_filtered // 2 + 1)
    xf_primeira_metade_filtered = xf_filtered[primeira_metade_indices_filtered]
    magnitude_espectro_filtered = 2.0/N_filtered * np.abs(yf_filtered[primeira_metade_indices_filtered])

    plt.subplot(2, 1, 2) # 2 linhas, 1 coluna, segundo gráfico
    plt.plot(xf_primeira_metade_filtered, magnitude_espectro_filtered, color='orange')
    plt.title(f'Espectro de Frequência Filtrado')
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Magnitude (Amplitude Normalizada)')
    plt.grid(True)
    plt.xscale('log')
    plt.xlim([20, sr/2]) # Faixa audível

    plt.tight_layout()
    plt.show()

    # --- 4. Salvar o Áudio Filtrado ---
    caminho_saida = os.path.join(os.path.dirname(caminho_arquivo_audio), nome_arquivo_saida)
    print(f"Salvando áudio filtrado como: {caminho_saida}...")
    try:
        # Normalizar o áudio antes de salvar para evitar clipping, se necessário
        # soundfile espera float64 por padrão, mas float32 é comum para áudio.
        # librosa.load geralmente retorna float32.
        # Garante que os valores estejam no intervalo [-1.0, 1.0]
        y_filtered_normalized = y_filtered / np.max(np.abs(y_filtered)) if np.max(np.abs(y_filtered)) > 1.0 else y_filtered
        
        # Use sf.write em vez de librosa.output.write_wav
        sf.write(caminho_saida, y_filtered_normalized, sr) # <-- Linha corrigida!
        print("Áudio filtrado salvo com sucesso!")
    except Exception as e:
        print(f"Erro ao salvar o áudio filtrado: {e}")
        print("Certifique-se de que a biblioteca 'soundfile' está instalada: pip install soundfile")

    print("Análise e filtragem concluídas.")


if __name__ == "__main__":
    
    musica = 'Dancing Alone.wav' 
    # Nome do arquivo de saída para o instrumental
    nome_do_arquivo_instrumental = 'exemplo_instrumental.wav'

    analisar_e_filtrar_audio(musica, nome_do_arquivo_instrumental)
