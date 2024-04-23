from transformers import (
    LlamaForCausalLM,
    LlamaConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoTokenizer
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  Subset
from random import sample
import json
from pathlib import Path
import wandb

from dataset.babylm.babylm_dataset import BabylmDataset

#############
LR = 2.5e-4
BATCH_SIZE = 8
SEQ_LENGTH = 128

TEMPERATURE = 2.0
ALPHA = 0.5

MODEL_PATH = Path("/home/kylin/models/")
STUDENT_NAME = f'Baby-Llama-58M'
TEACHER_NAME = f'Meta-Llama-3-8B'
MODEL_OUTPUT = Path('/home/kylin/models/') /  STUDENT_NAME
EVAL_SAMPLES = 8192
DATA_PATH = Path("/home/kylin/datasets")
wandb_log = True
#############

teacher_dir = MODEL_PATH / TEACHER_NAME

tokenizer = AutoTokenizer.from_pretrained(teacher_dir)

# in the original code I had random_chunk = False
# random_chunk=True is expected to improve the model performance a bit
train_dataset = BabylmDataset(DATA_PATH / "babylm_10M_clean", SEQ_LENGTH, tokenizer=tokenizer, random_chunk=True)
full_eval_dataset = BabylmDataset(DATA_PATH / "babylm_dev_clean", SEQ_LENGTH, tokenizer=tokenizer, offset=0)

eval_indices = sample(range(len(full_eval_dataset)), EVAL_SAMPLES)
eval_dataset = Subset(full_eval_dataset, eval_indices)

tokenizer.model_max_length = SEQ_LENGTH

with open('/home/kylin/workspace/llama-68m/config/Llama-3-74M.json', 'r') as file:
    config_dict = json.load(file)
    
config = LlamaConfig.from_dict(config_dict)
student = LlamaForCausalLM(config)
teacher = LlamaForCausalLM.from_pretrained(teacher_dir,torch_dtype=torch.bfloat16)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

print(f'model num parameters: student = {student.num_parameters()}')
print(f'model num parameters: teacher = {teacher.num_parameters()}')

#  Distillation Trainer
#  We modified the Trainer from this repo https://github.com/philschmid/knowledge-distillation-transformers-pytorch-sagemaker
# to work with an ensemble of teachers

class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self._move_model_to_device(self.teacher, self.model.device)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        # compute student output
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss

        # compute teacher output
        with torch.no_grad():
            outputs_teacher = teacher(**inputs)
            teacher_logits=outputs_teacher.logits

        # assert size
        # print("outputs_student.logits.size()",outputs_student.logits.size(),avg_teacher_logits.size())
        assert outputs_student.logits.size() == teacher_logits.size()

        # Soften probabilities and compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (
            loss_function(
                F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
                F.softmax(teacher_logits / self.args.temperature, dim=-1),
            )
            * (self.args.temperature ** 2)
        )
        # Return weighted student loss
        loss = self.args.alpha * student_loss + (1.0 - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss


if wandb_log:
    wandb.login()
    wandb.init(project='babylm', name=STUDENT_NAME)

training_args = DistillationTrainingArguments(
    output_dir=MODEL_OUTPUT,
    overwrite_output_dir=True,
    save_strategy = "epoch",
    evaluation_strategy = "epoch",
    num_train_epochs=6,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=BATCH_SIZE,
    save_total_limit=1,  # Set to zero to avoid saving
    report_to="wandb",
    warmup_steps=200, 
    lr_scheduler_type="cosine",
    learning_rate=LR,
    logging_steps=20,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    weight_decay=0.1,
    alpha=ALPHA,
    temperature=TEMPERATURE,
)

trainer = DistillationTrainer(
        student,
        training_args,
        teacher_model=teacher,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,

    )

trainer.train()

trainer.save_model(MODEL_OUTPUT)
tokenizer.save_pretrained(MODEL_OUTPUT)