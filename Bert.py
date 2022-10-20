from transformers import GPT2Model, BertConfig, GPT2Tokenizer, TrainingArguments, Trainer, get_scheduler, AutoTokenizer, DataCollatorWithPadding, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, PreTrainedModel, PretrainedConfig, AutoModel,T5ForConditionalGeneration
from datasets import load_dataset, load_metric, arrow_dataset, DatasetDict
from torch.optim import AdamW
from torch.utils.data import DataLoader
import os
from datasets.combine import concatenate_datasets
import numpy as np
import json
import torch
from tqdm.auto import tqdm
import pandas as pd
from nltk import ngrams

class Bert:
    # WordPiece is a bool same with use only lowercase should be(true, true) for first test
    def __init__(self, vocabFile, wordPiece, onlyLowercase):
        #might remove
        #del self.model
        #del tokenized_dataset
        # del self.trainer
        #self.DatasetReparser("OldDatasets/train.json", "train.json")
        print("done reparsing train.json")
        #self.DatasetReparser("OldDatasets/test.json", "test.json") 
        print("done reparsing test.json")
        torch.cuda.empty_cache()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        #self.configuration = BertConfig()
        """GPT2
        self.model = GPT2Model.from_pretrained("gpt2")
        self.model.to(torch.device("cpu"))
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        """
        datasetFiles = {"train": "train.json", "test": "test.json"}
        self.dataset = load_dataset("json", data_files=datasetFiles)
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        self.testmodel = AutoModelForSeq2SeqLM.from_pretrained("results/checkpoint-31000")
        #Collator below is for gpt2
        self.testmodel.config.max_length = 128
        #data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors='pt', max_length=1024, padding='max_length' )
        #Collator below is for seq2seq
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer,model=self.model)
        self.metric = load_metric("rouge")
        #self.tokenize_and_save_dataset()
        Tokenized_Dataset = DatasetDict.load_from_disk("Tokenized_datasets")
        #tokenized_dataset = dataset.map(self.tokenizer_function,batched=True,remove_columns=["input","text"])
        print(len(Tokenized_Dataset["train"]))
        print(len(Tokenized_Dataset["test"]))
        train_args = Seq2SeqTrainingArguments(output_dir="./results", evaluation_strategy="epoch",learning_rate=2e-5,per_device_train_batch_size=16,per_device_eval_batch_size=16, weight_decay=0.01, save_total_limit=3,num_train_epochs=1)

    
        #tokenized_dataset = tokenized_dataset.remove_columns(tokenized_dataset["train"].column_names)
                                         
        
        #tokenized_dataset.remove_columns(["input","text","input_ids","token_type_ids","attention_mask"])

        small_train_dataset = Tokenized_Dataset["train"].shuffle(seed=42).select(range(500000))
        #small_train_dataset = small_train_dataset.rename_column("labels","label")
        small_train_dataset.set_format("torch")
        #print(small_train_dataset[12])
        small_test_dataset = Tokenized_Dataset["test"].shuffle(seed=42).select(range(1496))
        self.trainer = Seq2SeqTrainer(model=self.model,args=train_args,train_dataset=small_train_dataset,eval_dataset=small_test_dataset,data_collator=data_collator,tokenizer=self.tokenizer)
        print(small_train_dataset)
        self.train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8,collate_fn=data_collator)
        self.test_dataset = self.dataset["test"].shuffle(seed=23).select(range(200))
        self.eval_dataloader = DataLoader(small_test_dataset, batch_size=8,collate_fn=data_collator)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(torch.cuda.is_available())
    def tokenize_and_save_dataset(self):
        
        #datasetFiles = {"train": "train.json", "test": "test.json"}
        #self.dataset = load_dataset("json", data_files=datasetFiles)
        print(self.dataset)
        tokenized_dataset = self.dataset.map(self.tokenizer_function,batched=True)
        
        #tokenized_dataset = tokenized_dataset.remove_columns(tokenized_dataset["train"].column_names)
        tokenized_dataset.save_to_disk("Tokenized_datasets")
        
    def train(self, num_epochs):
        self.trainer.train()
        """for GPT2 
        num_training_steps = num_epochs * len(self.train_dataloader)
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
        progress_bar = tqdm(range(num_training_steps))

        self.model.to(self.device)
        self.model.train()
        for epoch in range(num_epochs):
            for batch in self.train_dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                labels = {k: v.to(self.device) for k, v in batch.items()}
                inputs.pop("labels")
                del labels["input_ids"], labels["attention_mask"]
                print({k: v.shape for k,v in inputs.items()})
                print("........................")
                print({k: v.shape for k,v in labels.items()})
                outputs = self.model(**inputs)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
        """
    def test(self,inputs):

        #config = T5ForConditionalGeneration.from_pretrained("results/checkpoint-31000")
        #config.max_new_tokens = 128
        #model = config #PreTrainedModel(config)
        #print("done w/ config")
        #model.config.max_length = 128
        tokenizedInputs = self.tokenizer_function(inputs)
        #print(inputs["text"])
        rawOutput = self.testmodel.generate(tokenizedInputs["input_ids"])
        output = self.tokenizer.batch_decode(rawOutput,skip_special_tokens=True)
        #print(output[0])
        rawOutput = rawOutput.numpy()[0]
        outputNgram = output[0].split()
        inputsNgram = inputs["text"].split()
        #print(outputNgram)
        #result = self.metric.compute(predictions=outputNgram,references=inputsNgram)#.numpy()[0])
        return [inputsNgram,outputNgram]#[str(inputs["text"]),str(output[0])]
    
    def metricTest(self):
        sumVal = 0
        refs = [["hello how are you","hey, how you going"]]
        outputs = ["hello how are you doing today?"]
        print("starting Seq")
        for element in self.test_dataset.select(range(200)):
            #refs.append([self.test(element)[0],self.test(element)[0]])
           # outputs.append(self.test(element)[1])
            self.metric.add_batch(predictions=[self.test(element)[1]], references=[[self.test(element)[0]]])
        
        #print(refs)
        #print("---------------------------------------------------------------------------")
        #print(outputs)
        #self.metric.
        metric = self.metric.compute()#predictions=[outputs],referenences=[refs])
        print(metric)
    def printExample(self,inputs):
        print(inputs["input"])
        print("--------------------------------")
        tokenizedInputs = self.tokenizer_function(inputs)
        print(inputs["text"])
        rawOutput = self.testmodel.generate(tokenizedInputs["input_ids"])
        output = self.tokenizer.batch_decode(rawOutput,skip_special_tokens=True)
        print(output[0])
        
    def tokenizer_function(self, example):
    
        model_inputs = self.tokenizer(example["input"], max_length=128, padding='max_length',truncation=True, return_tensors="pt")
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(example["text"], max_length=128, padding='max_length', truncation=True,return_tensors="pt")
        #print(type(model_inputs)) 
        #model_input={'input_ids':model_inputs["input_ids"],'token_type_ids':model_inputs["token_type_ids"],"attention_mask":model_inputs["attention_mask"],"labels":labels["input_ids"]}
        model_inputs['labels'] = labels["input_ids"]
        #inputs=example["input"]
        #targets=example["text"]
        #model_inputs = self.tokenizer(inputs, text_target=targets,max_length=128, trunctaion=True)
        return model_inputs
        #print(type(self.tokenizer(example["input"],example["text"],truncation=True,padding=True,max_length=512)))
        #return self.tokenizer(example["input"],example["text"],truncation=True,padding=True,max_length=512)
        #tokens = self.tokenizer.tokenize(example)
        #ids = self.tokenizer.convert_tokens_to_ids(tokens)
        #final_inputs = self.tokenizer.prepare_for_model(ids)

    #Note that this is only for wiki bio
    def DatasetReparser(self, originalFileDir, NewFileDir):
        print("print")
        of=open(originalFileDir)
        nf=open(NewFileDir, 'w',encoding="utf-8")
        nf.seek(0)
        nf.truncate()
        with of as handle:
            json_data = [json.loads(line) for line in handle]
            print("done w/ json data")

        for unfilteredData in json_data:
            #print("in loop")
            data_buffer = []
            for element in unfilteredData['data']:
                data_buffer.append(' - '.join(element))
            data = "data: " + ' / '.join(data_buffer)
            docTitle = "document title: " + unfilteredData["doc_title"] + " | "
            docTitleBPE = "document title with bpe: " + unfilteredData["doc_title_bpe"] + " | "
            SecTitle = "section title: " + ' - '.join(unfilteredData["sec_title"]) + " | "
            text = json.dumps(unfilteredData["text"])
            finalInput = docTitle + docTitleBPE + SecTitle + data
            finalLine= "{\"input\":"+ json.dumps(finalInput).replace("@@ ","") + ", " + "\"text\":" + text.replace("@@ ","") +"}\n"
            nf.write(finalLine)
        #nf.truncate(50000)
