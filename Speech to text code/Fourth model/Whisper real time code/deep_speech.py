import os
import pyaudio
import wave
import numpy as np
import whisper

# Function to record audio from microphone
def record_audio(seconds, output_file):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000  # DeepSpeech model sample rate
    RECORD_SECONDS = seconds

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    frames = []

    print("Recording...")

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(output_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

# Load the whisper model
model = whisper.load_model("base.en")

while True:
    # Record audio from the microphone and save it to a file
    record_audio(seconds=3, output_file="output.wav")

    # Transcribe the recorded audio
    result = model.transcribe("output.wav")

    # Print the transcribed text
    transcribed_text = result["text"]
    print("Transcribed text:", transcribed_text)

    # Check if the transcribed text contains the word "exit"
    if "exit" in transcribed_text.lower():
        print("Exit command detected. Exiting...")
        break
