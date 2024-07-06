import os
from pydub import AudioSegment

def convert_sample_rate(input_folder, output_folder, target_sample_rate=16000):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):  
            file_path = os.path.join(input_folder, filename)
            audio = AudioSegment.from_file(file_path)
            audio = audio.set_frame_rate(target_sample_rate)
            
            output_path = os.path.join(output_folder, filename)
            audio.export(output_path, format="wav")  
            
            print(f"Converted {filename} to {target_sample_rate} Hz")

input_folder = "C:\\Users\\BIA\\Desktop\\data-set-ears\\blind_testset"  
output_folder = "C:\\Users\\BIA\Desktop\\data-set-ears\\blind-convert"  
convert_sample_rate(input_folder, output_folder)
