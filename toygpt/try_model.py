from model import GPT

def main():
    _ = GPT.from_pretrained('gpt2')

    print("Did't crash yay!")


if __name__ == '__main__':
    main()

