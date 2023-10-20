from beautiful_prompt.utils import read_json

data = read_json('to_pick_data.json')

data = [d for d in data if 'pick' in d]

methods = ['raw', 'magic-prompt', 'chatgpt', 'beautiful-prompt-sft']

win = [0, 0, 0, 0]
tie = [0, 0, 0, 0]
loss = [0, 0, 0, 0]
total = [0, 0, 0, 0]
for d in data:
    if d['method1'] in methods:
        index = methods.index(d['method1'])
        if d['pick'] == 'img1':
            loss[index] += 1
        elif d['pick'] == 'img2':
            win[index] += 1
        elif d['pick'] in ['pie', 'tie']:
            tie[index] += 1
        else:
            print(d['pick'])
            assert 0
    else:
        index = methods.index(d['method2'])
        if d['pick'] == 'img2':
            loss[index] += 1
        elif d['pick'] == 'img1':
            win[index] += 1
        elif d['pick'] in ['pie', 'tie']:
            tie[index] += 1
        else:
            print(d['pick'])
            assert 0

    total[index] += 1

print(win, tie, loss, total)
for i in range(len(methods)):
    print(methods[i])
    print(win[i] / total[i])
    print(tie[i] / total[i])
    print(loss[i] / total[i])
