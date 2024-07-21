
import sys
from modeling_gpt import GPT

def main():
    m = GPT.from_pretrained('gpt2')
    print("Did't crash yay!")

    print('testing generation')
    prompt = '''Building a multi-class classification model in PyTorch,
'''
    print(">> Prompt:\n", prompt, "\n>> Completion:")
    for i in m.generate(prompt, num_return_tokens=64, sampling=True, topk=50, temperature=0.4):
        sys.stdout.write(i)
        sys.stdout.flush()
    return

if __name__ == '__main__':
    main()

