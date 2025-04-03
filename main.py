import os
import subprocess
import whisper

def download_audio(youtube_url, output_filename="downloaded_audio.wav"):
    """
    Download the best audio from the YouTube URL as a WAV file.
    """
    download_cmd = [
        "yt-dlp",
        "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", "downloaded_audio.%(ext)s",
        youtube_url
    ]
    print("Downloading audio from YouTube...")
    subprocess.run(download_cmd, check=True)
    if not os.path.exists(output_filename):
        raise FileNotFoundError(f"{output_filename} was not created.")
    print("Download complete.")
    return output_filename

def convert_audio(input_file, output_file="converted_audio.wav", sample_rate="22050"):
    """
    Convert audio to 22050Hz, mono, 16-bit PCM using ffmpeg.
    """
    convert_cmd = [
        "ffmpeg",
        "-i", input_file,
        "-ar", sample_rate,
        "-ac", "1",
        "-sample_fmt", "s16",
        output_file,
        "-y"  # Overwrite output if exists
    ]
    print("Converting audio to 22050Hz, mono, 16-bit...")
    subprocess.run(convert_cmd, check=True)
    print("Conversion complete.")
    return output_file

def transcribe_audio(audio_file):
    """
    Use Whisper to transcribe the audio.
    Returns a list of segments; if no segments are provided, returns one segment.
    Each segment is a dict with keys: "start", "end", and "text".
    """
    print("Transcribing audio with Whisper (this may take a while)...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    segments = result.get("segments")
    if segments is None or len(segments) == 0:
        # Fallback: treat entire transcription as one segment
        segments = [{"start": 0, "end": None, "text": result.get("text", "").strip()}]
    print("Transcription complete.")
    return segments

def extract_segment(audio_file, start, end, output_path):
    """
    Use ffmpeg to extract a segment from the audio file.
    """
    if end is not None:
        extract_cmd = [
            "ffmpeg",
            "-i", audio_file,
            "-ss", str(start),
            "-to", str(end),
            "-c", "copy",
            output_path,
            "-y"
        ]
    else:
        extract_cmd = [
            "ffmpeg",
            "-i", audio_file,
            "-ss", str(start),
            "-c", "copy",
            output_path,
            "-y"
        ]
    subprocess.run(extract_cmd, check=True)

def create_dataset_structure(segments, source_audio, dataset_dir="dataset"):
    """
    Creates the dataset folder structure, extracts segments to wav files, and writes metadata.csv.
    """
    wavs_dir = os.path.join(dataset_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)
    
    metadata_lines = []
    file_counter = 1

    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"].strip()
        wav_filename = f"{file_counter:04d}.wav"
        output_path = os.path.join(wavs_dir, wav_filename)
        print(f"Extracting segment {file_counter}: {wav_filename} (from {start}s to {end if end is not None else 'end'})")
        extract_segment(source_audio, start, end, output_path)
        metadata_lines.append(f"{wav_filename}|{text}")
        file_counter += 1

    metadata_path = os.path.join(dataset_dir, "metadata.csv")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_lines))
    print(f"Metadata file created at {metadata_path}.")

def main():
    youtube_url = input("Enter the YouTube video URL: ").strip()

    # Step 1: Download audio
    downloaded_file = download_audio(youtube_url)

    # Step 2: Convert audio to the required format
    converted_file = convert_audio(downloaded_file)

    # Step 3: Transcribe audio with Whisper
    segments = transcribe_audio(converted_file)

    # Step 4: Create dataset folder structure and extract segments
    create_dataset_structure(segments, converted_file)

    print("Dataset creation complete. Check the 'dataset' folder.")

if __name__ == "__main__":
    main()