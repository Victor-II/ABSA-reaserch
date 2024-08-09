from typing import List
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from copy import deepcopy

class DS(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_len: int=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.inputs = inputs
        self.targets = targets

        self.inputs_out = []
        self.targets_out = []

        self._build_data()

    def __len__(self):
        return len(self.inputs)
    
    def _build_data(self):

        for input, target in zip(self.inputs, self.targets):

            input = ' '.join(input)
            target = ' '.join(target)

            tokenized_input = self.tokenizer(
              input, 
              max_length=self.max_len, 
              padding='max_length', 
              truncation=True,
              return_tensors="pt",
            )
            tokenized_target = self.tokenizer(
              target, 
              max_length=self.max_len, 
              padding='max_length', 
              truncation=True,
              return_tensors="pt"
            )

            self.inputs_out.append(tokenized_input)
            self.targets_out.append(tokenized_target)
    
    def __getitem__(self, index) -> dict:
        return {
            'input_ids': self.inputs_out[index].input_ids,
            'attention_mask': self.inputs_out[index].attention_mask,
            'labels': self.targets_out[index].input_ids,
            'decoder_attention_mask': self.targets_out[index].attention_mask
        }
        
class DataProcessor ():

    def __init__(self, source: str='silviolima'):
        self.preprocessed = dict()

        if source == 'silviolima':
            self.preprocess_silviolima()
        else:
            print('-> Unsupported data source')

    def preprocess_silviolima (self) -> tuple[list[list[str]], list[list[list[str]]]]:
        data = load_dataset('SilvioLima/absa')
        s2w = { 'NEG': 'negative',
                'POS': 'positive',
                'NEU': 'neutral' }

        for split in ['train', 'valid', 'test']:
            inputs = []
            targets = []
            for item in data[split]:
                inputs.append(item['sentence'].split())
                target = []
                for t in eval(item['triples']):
                    t = list(t)
                    t[2] = s2w[t[2]]
                    if t[0] == -1: t[0] = 'NULL'
                    target.append(t)
                targets.append(target)

            self.preprocessed[split] = inputs, targets

        return self.preprocessed     
    
    def _processing_step (self, input: list[str], target: list[list[str]], task: str, policy: str) -> tuple[list[str], list[str]]:
        input_out = []
        target_out = []
        label_idx = []

        if 'a' in task: label_idx.append(0) 
        if 'o' in task: label_idx.append(1) 
        if 'p' in task: label_idx.append(2) 

        if policy == 'base':
            input_out = input

        elif policy == 'functional':
            funcs = ['<ASPECT>', '<OPINION>', '<POLATITY>']
            input_out.extend([funcs[i] for i in label_idx])
            input_out.extend(input)

        target_out.append('[')
        for t in target:
            t_out = []
            t_out.append('[')
            t_out.append(t[label_idx[0]])

            if len(label_idx) > 1: 
                t_out.append('|')
                t_out.append(t[label_idx[1]])
            
                if len(label_idx) > 2:
                    t_out.append('|')
                    t_out.append(t[label_idx[2]])

            t_out.append(']')
            target_out.extend(t_out)
        target_out.append(']')

        return input_out, target_out
    

    def process (self, split: str, policy: str='base', task: str='aop') -> tuple[list[list[str]],  list[list[str]]]:
        inputs = []
        targets = []

        for input, target in zip(*self.preprocessed[split]):
            i, t = self._processing_step(input, target, task, policy)
            inputs.append(i)
            targets.append(t)
        
        return inputs, targets

if __name__ == '__main__':
    
    dp = DataProcessor()
    inputs, targets = dp.process(split='valid', task='poa', policy='base')

    tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small')
    ds = DS(inputs, targets, tokenizer)

    print(ds[0])

