import torch
import pytorch_lightning as pl
import json
import numpy as np
import editdistance
import os
import re
import logging
from tqdm import tqdm
from argparse import ArgumentParser
from pytorch_lightning import seed_everything
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_linear_schedule_with_warmup

LOGGER = logging.getLogger(__name__)

def init_config():
    parser = ArgumentParser()
    parser.add_argument('--cfg', default='default', help='Config to use. Path relative to config dir', type=str)
    parser.add_argument('--policy', type=str)
    parser.add_argument('--task', type=str)

    parser.add_argument('--mode', type=str)
    parser.add_argument('--model_ckpt', type=str)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--train_batch_size', type=int)
    parser.add_argument('--eval_batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--epsilon', type=float)
    parser.add_argument('--warmup_steps', type=int)
    parser.add_argument('--grad_accumulation_steps', type=int)
    parser.add_argument('--gradient_clip_val', type=float)

    args = parser.parse_args()
    cfg = get_config(args.cfg)
    for k, v in cfg.items():
        if vars(args).get(k, None) != None and vars(args)[k] != v:
            cfg[k] = vars(args)[k]

    return cfg

def get_config(cfg: str):
    if len(cfg.split('.')) > 1 and cfg.split('.')[1] == 'json':
        path = f'../config/{cfg}'
    else:
        path = f'../config/{cfg}.json'

    with open(path) as c:
        config = json.load(c)
    return config
    
        
class DataProcessor():

    def __init__(self, source: str, pmap: dict, fmap: dict, seed: int):
        self.preprocessed = dict()
        self.pmap = pmap
        self.fmap = fmap
        self.seed = seed

        if source == 'silviolima':
            self.preprocess_silviolima()
        else:
            print('-> Unsupported data source')

    def preprocess_silviolima (self) -> tuple[list[list[str]], list[list[list[str]]]]:
        data = load_dataset('SilvioLima/absa')

        for split in ['train', 'valid', 'test']:
            inputs = []
            targets = []
            for item in data[split]:
                inputs.append(item['sentence'].split())
                target = []
                for t in eval(item['triples']):
                    t = list(t)
                    t[2] = self.pmap[t[2]]
                    if t[0] == -1: t[0] = 'NULL'
                    target.append(t)
                targets.append(target)

            self.preprocessed[split] = inputs, targets
            self.preprocessed[split] = list(zip(*self.preprocessed[split]))

    def _processing_step (self, input: list[str], target: list[list[str]], task: str, policy: str) -> tuple[list[str], list[str]]:
        input_out = []
        target_out = []
        fidMap = { 'a': 0, 'o': 1, 'p': 2}

        if policy == 'base':
            input_out = input

        elif policy == 'functional' or policy == 'compositional':
            try:
                input_out.extend([self.fmap[f] for f in task])
            except:
                print('-> ERR: "task" param cannot contain more than one of each letters: "a", "o", "p"')
                return
            input_out.extend(input)

        for t in target:
            t_out = []
            t_out.append('[')
            for f in task:
                try:
                    t_out.append(t[fidMap[f]])
                except:
                    print('-> ERR: "task" param cannot contain more than one of each letters: "a", "o", "p"')
                    return
                t_out.append('|')
            t_out.pop()
            t_out.append(']')
            target_out.extend(t_out)

        return input_out, target_out
    

    def process (self, split: str, policy: str, task: str, partitions_split: list[float]) -> tuple[list[list[str]],  list[list[str]]]:
        inputs = []
        targets = []
        tasks = []
        if policy == 'base' or policy == 'functional':
            for input, target in self.preprocessed[split]:
                i, t = self._processing_step(input, target, task, policy)
                inputs.append(i)
                targets.append(t)
                tasks.append(task)

        # TODO: Find way for this to work with all tasks, and not hard code it for 'aop' only
        elif policy == 'compositional':
            assert len(task) == 3, 'the "compositional" policy supports only the "aop" task'
            rng_shuffle = np.random.default_rng(self.seed)
            n = len(self.preprocessed[split])
            rng_shuffle.shuffle(self.preprocessed[split])
            p1 = self.preprocessed[split][ 0                                                     : int(n*partitions_split[0])                            ]
            p2 = self.preprocessed[split][ int(n*partitions_split[0])                            : int(n*partitions_split[0])+int(n*partitions_split[1]) ]
            p3 = self.preprocessed[split][ int(n*partitions_split[0])+int(n*partitions_split[1]) :                                                       ]

            rng_choice = np.random.default_rng()
            for input, target in p1:
                _task = rng_choice.choice(['a', 'o', 'p'])
                i, t = self._processing_step(input, target, _task, policy)
                inputs.append(i)
                targets.append(t)
                tasks.append(_task)

            for input, target in p2:
                _task = rng_choice.choice(['ao', 'oa', 'ap', 'pa', 'op', 'po'])
                i, t = self._processing_step(input, target, _task, policy)
                inputs.append(i)
                targets.append(t)
                tasks.append(_task)


            for input, target in p3:
                _task = rng_choice.choice(['aop', 'apo', 'oap', 'opa', 'pao', 'poa'])
                i, t = self._processing_step(input, target, _task, policy)
                inputs.append(i)
                targets.append(t)
                tasks.append(_task)

        return inputs, targets, tasks
    
class DataEncoder(Dataset):
    def __init__(self, inputs, targets, tasks, tokenizer, max_len: int=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.inputs = inputs
        self.targets = targets
        self.tasks = tasks

        self.inputs_out = []
        self.targets_out = []

        self.encode_data()

    def __len__(self):
        return len(self.inputs)
    
    def encode_data(self):

        for input, target in zip(self.inputs, self.targets):

            input = ' '.join(input)
            target = ' '.join(target)

            tokenized_input = self.tokenizer(
              [input], 
              max_length=self.max_len, 
              padding='max_length', 
              truncation=True,
              return_tensors="pt",
            )

            tokenized_target = self.tokenizer(
              [target], 
              max_length=self.max_len, 
              padding='max_length', 
              truncation=True,
              return_tensors="pt"
            )

            self.inputs_out.append(tokenized_input)
            self.targets_out.append(tokenized_target)
            
    def __getitem__(self, index) -> dict:
        return {
            'input_ids': self.inputs_out[index]['input_ids'].squeeze(),
            'attention_mask': self.inputs_out[index]['attention_mask'].squeeze(),
            'decoder_input_ids': self.targets_out[index]['input_ids'].squeeze(),
            'decoder_attention_mask': self.targets_out[index]['attention_mask'].squeeze(),
            'task': self.tasks[index]
            }

class DataModule():
    def __init__(self, source: str, pmap: dict, fmap: dict, tokenizer_ckpt: str, max_len: int, seed: int) -> None:
        self.source = source
        self.pmap = pmap
        self.fmap = fmap
        self.tokenizer_ckpt = tokenizer_ckpt
        self.max_len = max_len
        self.seed = seed
        self.dp = self.build_processor()
        self.tokenizer = self.build_tokenizer()

    def build_tokenizer(self):
        tokenizer = T5Tokenizer.from_pretrained(self.tokenizer_ckpt, legacy=False, additional_special_tokens=[v for v in self.fmap.values()])
        # tokenizer.add_tokens([v for v in fmap.values()])
        return tokenizer
    
    def build_processor(self):
        return DataProcessor(self.source, self.pmap, self.fmap, self.seed)
    
    def get_dataset(self, split, policy, task, partitions_split):
        return DataEncoder(*self.dp.process(split, policy, task, partitions_split), self.tokenizer, self.max_len)
    
    def get_dataloader(self, split, batch_size, policy, task, partitions_split):
        return DataLoader(self.get_dataset(split, policy, task, partitions_split), batch_size=batch_size, shuffle=(split == 'train'))


class T5FineTuner(pl.LightningModule):
    def __init__(self, model_ckpt: str | None=None):
        super(T5FineTuner, self).__init__()
        
        if model_ckpt != None:
            self.model = T5ForConditionalGeneration.from_pretrained(model_ckpt)
        else:
            self.model = None
        self.history = {'train_loss': [], 'val_loss': []}

        self.train_dl = None
        self.val_dl = None
        self.test_dl = None

    def init_training_args(self,
                           learning_rate: float, 
                           weight_decay: float, 
                           epsilon: float,
                           warmup_steps: int,
                           num_training_steps: int):
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.warmup_steps = warmup_steps
        self.num_training_steps = num_training_steps

    def is_logger(self):
        return True

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, 
                decoder_attention_mask=None, labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        lm_labels = batch['decoder_input_ids']
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            decoder_input_ids = batch['decoder_input_ids'],
            decoder_attention_mask=batch['decoder_attention_mask'],
            labels=lm_labels,
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        logs = {"train_loss": loss}
        self.history['train_loss'].append(loss)
        return {"loss": loss, "log": logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logs = {"avg_train_loss": avg_train_loss}
        # self.history['train_loss'].append(avg_train_loss)
        return {"avg_train_loss": avg_train_loss, "log": logs, 'progress_bar': logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.history['val_loss'].append(loss)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_val_loss}
        # self.history['val_loss'].append(avg_val_loss)
        return {"avg_val_loss": avg_val_loss, "log": logs, 'progress_bar': logs}

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.epsilon)
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.warmup_steps, num_training_steps=self.num_training_steps
        )
        self.lr_scheduler = scheduler
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def set_dataloaders(self, train_dl: DataLoader=None, val_dl: DataLoader=None, test_dl: DataLoader=None):
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl

    def train_dataloader(self):
        assert self.train_dl != None, 'train_dl must be set using the set_dataloader() method'
        return self.train_dl
        
    def val_dataloader(self):
        assert self.train_dl != None, 'val_dl must be set using the set_dataloader() method'
        return self.val_dl
    
    def test_dataloader(self):
        assert self.train_dl != None, 'test_dl must be set using the set_dataloader() method'
        return self.test_dl
    
    def generate(self, kwargs):
        if self.model != None:
            return self.model.generate(**kwargs)

class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        LOGGER.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                LOGGER.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        LOGGER.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    LOGGER.info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))
    
class ModelEvaluator():
    def __init__(self, model_ckpt: str, tokenizer: T5Tokenizer, device: str) -> None:
        self.model = T5FineTuner()
        self.model.model = torch.load(model_ckpt)
        self.model.model.to(torch.device(device))
        self.model.model.eval()
        self.tokenizer = tokenizer

    def align_with_levenshtein_distance(term: str, original_sentece: list[str]) -> str:
        # source: https://github.com/NJUNLP/DMASTE/blob/main/Generative-ABSA/eval_utils.py
        term = term.split(' ')
        term_out = []

        for t in term:
            levenshtein_distance = []
            for w in original_sentece:
                ld = editdistance.eval(t, w)
                levenshtein_distance.append(ld)
            idx = levenshtein_distance.index(min(levenshtein_distance))
            term_out.append(original_sentece[idx])

        return ' '.join(term_out)

    def get_scores(self, predictions: list[str], targets: list[str], sentences: list[str], tasks: list[str]):
        # TODO: Make it work also when the task is always 'aop' (for the base experiment), and extract results for every subtask
        scores_raw = { 'a'  : {}, 'o'  : {}, 'p'  : {}, 'aop': {} }
        scores_aligned = { 'a'  : {}, 'o'  : {}, 'p'  : {}, 'aop': {} }

        a_raw   = {'predictions': [], 'targets': []}
        o_raw   = {'predictions': [], 'targets': []}
        p_raw   = {'predictions': [], 'targets': []}
        aop_raw = {'predictions': [], 'targets': []}

        a_aligned   = {'predictions': [], 'targets': []}
        o_aligned   = {'predictions': [], 'targets': []}
        p_aligned   = {'predictions': [], 'targets': []}
        aop_aligned = {'predictions': [], 'targets': []}

        predictions_raw:     list[list[list[str]]]  = [self.process(pred, task) for pred, task in zip(predictions, tasks)]
        predictions_aligned: list[list[list[str]]]  = [self.process(self.align_with_levenshtein_distance(pred), task) for pred, task in zip(predictions, tasks)]
        
        targets:             list[list[list[str]]]  = [self.process(target, task) for target, task in zip(targets, tasks)]

        for i, task in enumerate(tasks):
            if 'a' in list(task):
                j = list(task).index('a')
                a_raw['predictions'].append([[p[j]] for p in predictions_raw[i]])
                a_raw['targets'].append([[t[j]] for t in targets[i]])

                a_aligned['predictions'].append([[p[j]] for p in predictions_aligned[i]])
                a_aligned['targets'].append([[t[j]] for t in targets[i]])

            if 'o' in list(task):
                j = list(task).index('o')
                o_raw['predictions'].append([[p[j]] for p in predictions_raw[i]])
                o_raw['targets'].append([[t[j]] for t in targets[i]])

                o_aligned['predictions'].append([[p[j]] for p in predictions_aligned[i]])
                o_aligned['targets'].append([[t[j]] for t in targets[i]])

            if 'p' in list(task):
                j = list(task).index('p')
                p_raw['predictions'].append([[p[j]] for p in predictions_raw[i]])
                p_raw['targets'].append([[t[j]] for t in targets[i]])

                p_aligned['predictions'].append([[p[j]] for p in predictions_aligned[i]])
                p_aligned['targets'].append([[t[j]] for t in targets[i]])

            if 'a' in list(task) and 'o' in list(task) and 'p' in list(task):
                a_raw['predictions'].append([[p[j]] for p in predictions_raw[i]])
                a_raw['targets'].append([[t[j]] for t in targets[i]])

                aop_aligned['predictions'].append(predictions_aligned[i])
                aop_aligned['targets'].append(targets[i])
        
        scores_raw['a']   = self.compute_scores(**a_raw)
        scores_raw['o']   = self.compute_scores(**o_raw)
        scores_raw['p']   = self.compute_scores(**p_raw)
        scores_raw['aop'] = self.compute_scores(**aop_raw)

        scores_aligned['a']   = self.compute_scores(**a_aligned)
        scores_aligned['o']   = self.compute_scores(**o_aligned)
        scores_aligned['p']   = self.compute_scores(**p_aligned)
        scores_aligned['aop'] = self.compute_scores(**aop_aligned)

        return scores_raw, scores_aligned
    
    def compute_scores(self, predictions: list[list[list[str]]], targets: list[list[list[str]]]):
        # source: https://github.com/NJUNLP/DMASTE/blob/main/Generative-ABSA/eval_utils.py
        if len(predictions) == 0 or len(targets) == 0:
            return None
        
        n_tp, n_gold, n_pred = 0, 0, 0

        for i in range(len(predictions)):
            n_gold += len(targets[i])
            n_pred += len(predictions[i])

            for p in predictions[i]:
                if p in targets[i]:
                    n_tp += 1

        precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
        recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
        scores = {'precision': precision, 'recall': recall, 'f1': f1}

        return scores

    def process(prediction: str, task: str) -> list[dict]:
        pairs = [pair[1: -1] for pair in re.findall('\[.*?\]', prediction)]
        _pairs = []
        for pair in pairs:
            _pair = [] # { t:None for t in task}
            pair = pair.split('|')
            if len(pair) == len(task):
                for i, t in enumerate(list(task)):
                    _pair[i] = pair[i] # _pair[t] = pair[i]
            _pairs.append(_pair)
        return _pairs


    def evaluate(self, dl: DataLoader, policy: str):
        predictions = []
        targets = []
        sentences = []
        for batch in tqdm(dl):
            preds = self.model.generate({
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask'],
                'max_length': 128
            })

            predictions.extend(
                [self.tokenizer.decode(input_ids, skip_special_tokens=True) for input_ids in preds]
            )
            targets.extend(
                [self.tokenizer.decode(input_ids, skip_special_tokens=True) for input_ids in batch['decoder_input_ids']]
            )
            sentences.extend(
                [self.tokenizer.decode(input_ids, skip_special_tokens=True) for input_ids in batch['input_ids']]
            )
            tasks = batch['task']
        
        scores = self.get_scores(predictions, targets, sentences, tasks)

        return scores

def inference(model: T5ForConditionalGeneration, 
              tokenizer: T5Tokenizer,
              input: str,
              additional_tokens: list[str]):
    pass

def run(config: dict):
    CFG = init_config()
#     {   
#     "pmap": {"NEG": "negative", "POS": "positive", "NEU": "neutral"},
#     "fmap": {"a": "<ASPECT>", "o": "<OPINION>", "p": "POLARITY"},
#     "dataset": "silviolima",
#     "policy": "base",
#     "task": "aop",

#     "seed": 3141519,
#     "mode": "train-eval",
#     "model_ckpt": "google/t5-v1_1-small",
#     "epochs": 10,
#     "train_batch_size": "128",
#     "eval_batch_size": 128,
#     "learning_rate": 0.001,
#     "weight_decay": 0.0,
#     "epsilon": 1e-8,
#     "warmup_steps": 0,
#     "grad_accumulation_steps": 1,
#     "gpus": 2,
#     "accelerator": "gpu",
#     "gradient_clip_val": 1.0,
#     "max_seq_len": 256,

#     "dir_ckpt": "../checkpoints", 
#     "dir_exp": "../experiments"
# }
    if CFG['mode'] == 'train-eval' or CFG['mode'] == 'train':
        pl.seed_everything(CFG['seed'])

        dm = DataModule(
            source=CFG['source'],
            pmap=CFG['pmap'],
            fmap=CFG['fmap'],
            tokenizer_ckpt=CFG['tokeizer_ckpt'],
            max_len=CFG['max_seq_len'],
            seed=CFG['seed']
        )

        num_training_steps = (
            (len(dm.get_dataset(split='train', policy=CFG['policy'], task=CFG['task'], partitions_split=CFG['partitions_split'])) 
            // (CFG['train_batch_size'] * max(1, CFG['gpus'])))
            // CFG['grad_accumulation_steps'] * float(CFG['epochs'])
        )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=CFG['dir_ckpt'], prefix="ckt", monitor='val_loss', mode='min', save_top_k=3
        )

        model = T5FineTuner(CFG['model_ckpt'])
        model.init_training_args(
            learning_rate=CFG['learning_rate'],
            weight_decay=CFG['weight_decay'],
            epsilon=CFG['epsilon'],
            warmup_steps=CFG['warmup_steps'],
            num_training_steps=num_training_steps
        )
        model.set_dataloaders(
            train_dl=dm.get_dataloader(split='train', batch_size=CFG['train_batch_size'], policy=CFG['policy'], task=CFG['task'], partitions_split=CFG['partitions_split']),
            val_dl=dm.get_dataloader(split='val', batch_size=CFG['val_batch_size'], policy=CFG['policy'], task=CFG['task'], partitions_split=CFG['partitions_split']),
            test_dl=dm.get_dataloader(split='test', batch_size=CFG['val_batch_size'], policy=CFG['policy'], task=CFG['task'], partitions_split=CFG['partitions_split']),
        )

        train_params = dict(
            default_root_dir=os.path.join(CFG['dir_ckpt'], CFG['model_name']),
            accumulate_grad_batches=CFG['grad_accumulation_steps'],
            gpus=CFG['gpus'],
            gradient_clip_val=CFG['gradient_clip_val'],
            #amp_level='O1',
            max_epochs=CFG['epochs'],
            checkpoint_callback=checkpoint_callback,
            callbacks=[LoggingCallback()],
        )

        trainer = pl.Trainer(**train_params)
        trainer.fit(model)
        torch.save(model.model, os.path.join(CFG['dir_ckpt'], f'{CFG['model_name']}.pt'))
        torch.save(model.history, os.path.join(CFG['dir_exp'], CFG['exp_name'], f'h_{CFG['model_name']}.pt'))

    if CFG['mode'] == 'train-eval':
        pass


    


if __name__ == '__main__':
    checkpoint = 'google/t5-v1_1-small'
    pmap = { 'NEG': 'negative',
             'POS': 'positive',
             'NEU': 'neutral' }
    
    fmap = { 'a': '<ASPECT>',
             'o': '<OPINION>',
             'p': '<POLARITY>'}
    
    # dm = DataModule('silviolima', pmap, fmap, checkpoint, 128)
    # print(dm.tokenizer.encode('<ASPECT>'))
    # print(dm.tokenizer.encode('<OPINION>'))
    # print(dm.tokenizer.encode('<POLARITY>'))
    # print(dm.tokenizer.decode([32000, 32001, 32002, 456, 7584, 1], skip_special_tokens=False))
    # val_dl = dm.get_dataloader('valid', 1, policy='functional', task='aop')
    # print(len(val_dl))

    
    # cfg = init_config()
    # print(cfg)
    dm = DataModule('silviolima', pmap, fmap, checkpoint, 128, 3141519)
    print(dm.tokenizer.encode('['))
    print(dm.tokenizer.encode(']'))
    print(dm.tokenizer.encode('|'))

    # for i in range(10):
    #     print(dp.preprocessed['valid'][i])
    data = dm.get_dataloader('valid', batch_size=3, policy='compositional', task='aop', partitions_split=[0.5, 0.25, 0.25])

    # rng = np.random.default_rng()
    # idx = rng.choice(len(data[0][0]), 10)
    for item in data:
        print(f'{item['input_ids'] = }')
        print(f'{item['decoder_input_ids'] = }')
        print(f'{item['task'] = }')
        print('-'*50)
        break
    

