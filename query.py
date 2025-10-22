import torch

def main():
    try:
        if torch.cuda.is_available():
            print("CUDA available:", True)
            print("GPU name:", torch.cuda.get_device_name(0))
        else:
            print("CUDA available:", False)
            print("No GPU detected.")
        
        # Your main code logic goes here
        pass
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
