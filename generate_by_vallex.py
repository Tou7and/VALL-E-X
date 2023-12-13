"""
Generate speech from text prompt by VALL-E-X.

2023.10.31.
"""
import os
from glob import glob
from tqdm import tqdm
from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from utils.prompt_making import make_prompt

preload_models()

def infer_mix(text_prompt, output_file):
    # text_prompt="[EN] Engerix B [EN] [ZH] 免費追加 第一劑 [ZH]"

    audio_array = generate_audio(text_prompt, language='mix')
    write_wav(output_file, SAMPLE_RATE, audio_array)
    return

def voice_clone():
    make_prompt(
        name="chen2", 
        audio_prompt_path="audio_data/ntu-chen-common-3sec-24k.wav",
        transcript="除了 剛剛提到的 書籍資料 和 論壇資料")

    text_prompt = "金麟 起是 池中物，一遇 風雲 便化龍。"
    audio_array = generate_audio(text_prompt, prompt="chen2")
    write_wav("chen2_golddragon2.wav", SAMPLE_RATE, audio_array)

    return


def make_drug_dataset():
    prompt_files = glob("/home/t36668/projects/cmuh-ghi-dataset/data/synthetic_data/drug_name/data/vallex_prompts/*.txt")
    print(len(prompt_files))

    for prompt_file in tqdm(prompt_files):
        with open(prompt_file, "r") as reader:
            prompt = reader.read()
        base_name = os.path.basename(prompt_file).replace(".txt", ".wav")
        output_file = os.path.join("exp/drug_tts_corpus", base_name)

        if os.path.exists(output_file):
            print("already done!")
            continue

        infer_mix(prompt, output_file)
    
make_drug_dataset()
