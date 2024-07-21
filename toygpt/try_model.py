from model import GPT

def main():
    m = GPT.from_pretrained('gpt2')

    print("Did't crash yay!")
    print("m model parameters")

    m.generate('hello world!', num_return_tokens=32)

if __name__ == '__main__':
    main()

