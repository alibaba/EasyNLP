import os, json

file_path = '/home/moming/data/few_shot/efl_fewshot/cnewsum/10candi/'

names = os.listdir(file_path)
for n in names:
    lines = open(os.path.join(file_path, n), 'r').readlines()
    contents = [json.loads(l) for l in lines]
    for idx, c in enumerate(contents):
        new = []
        test_id = c['text_id']
        summary_id = c['summary_id']
        candidate_ids = c['candidate_id']
        
        trun_summary_id = summary_id[:180]
        summary = trun_summary_id + test_id[:512-len(trun_summary_id)]
        for cand in candidate_ids:
            trun_cand = cand[:200]
            new.append(trun_cand+test_id[:512-len(trun_cand)])
        contents[idx]['summary_id'] = summary
        contents[idx]['candidate_id'] = new
    for c in contents:
        with open(os.path.join(file_path, 'concat_'+n), 'a') as f:
            print(json.dumps(c), file=f)

