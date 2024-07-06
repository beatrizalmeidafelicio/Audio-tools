import os
import argparse
import librosa
from tqdm import tqdm
from glob import glob

def convert_file(input_filepath, output_filepath):
    y, sr = librosa.load(input_filepath, sr=16000)  # Carrega o arquivo MP3 e resample para 16 kHz
    librosa.output.write_wav(output_filepath, y, sr, norm=False)  # Salva como WAV PCM16

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='input', help='Pasta de entrada')
    parser.add_argument('-o', '--output', default='output', help='Pasta de saída')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    for input_filepath in tqdm(glob(os.path.join(args.input, '*.mp3'))):
        filename = os.path.splitext(os.path.basename(input_filepath))[0]
        output_filepath = os.path.join(args.output, f'{filename}.wav')
        convert_file(input_filepath, output_filepath)

if __name__ == "__main__":
    main()
