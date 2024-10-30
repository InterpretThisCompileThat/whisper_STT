import argparse
import whisper

def transcribe_audio(file_path, model_name="base", verbose=False):
    # Load the Whisper model
    if verbose:
        print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)

    # Perform transcription
    if verbose:
        print(f"Transcribing audio file: {file_path}")
    result = model.transcribe(file_path)

    # Print the transcription result
    print("Transcription:")
    print(result["text"])

def main():
    parser = argparse.ArgumentParser(description="Speech-to-Text CLI Tool using Whisper")
    parser.add_argument("file", type=str, help="Path to the audio file to transcribe")
    parser.add_argument("--model", type=str, default="base", choices=["tiny", "base", "small", "medium", "large"], help="Whisper model to use for transcription (default: base)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Call the transcription function with the parsed arguments
    transcribe_audio(args.file, model_name=args.model, verbose=args.verbose)

if __name__ == "__main__":
    main()
