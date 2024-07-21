
import sys
from toygpt.modeling_gpt import GPT

def main():
    m = GPT.from_pretrained('gpt2')
    print("Did't crash yay!")

    prompt = 'Romeo, oh Romeo, why is thy name'

    print(prompt)
    for i in m.generate(prompt, num_return_seq=1, num_return_tokens=64):
        sys.stdout.write(i)
        sys.stdout.flush()

if __name__ == '__main__':
    main()

