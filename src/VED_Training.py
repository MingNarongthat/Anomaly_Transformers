import torch
import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
import random
from PIL import Image
import json

# Decoder
from transformers import default_data_collator

#Encoder-Decoder Model
from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTFeatureExtractor

#Training
from transformers import Trainer, TrainingArguments

# If there's a GPU available
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

images_path = "/opt/project/dataset/DataAll/Training/"
train_test_ratio = 0.1

# Load the pre-trained image captioning model and tokenizer
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained\
                                    ("google/vit-base-patch16-224-in21k", 'bert-base-uncased', tie_encoder_decoder=True)
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Read json file containing the image name and captioning
with open("/opt/project/dataset/caption_dataset_normal.json", 'r') as f:
    datastore = json.load(f)

# Preparing training and testing set
image = []
caption = []
for item in datastore:
    image.append(images_path+item['image'])
    caption.append(item['caption'])

# convert to array in order to shuffle the data
image = np.array(image)
caption = np.array(caption)
indices = np.arange(image.shape[0])
random.shuffle(indices)

# after shuffling convert to list again
image = np.array(image[indices]).tolist()
caption = np.array(caption[indices]).tolist()

# select the amount of data by using ratio that we have set. e.g. 0.2
training_size = round(len(image)*(1-train_test_ratio))

training_image = image[0:training_size]
testing_image = image[training_size:]
training_caption = caption[0:training_size]
testing_caption = caption[training_size:]

# convert to dataframe for input in the model
train_df = pd.DataFrame({'images': training_image, 'captions': training_caption})
test_df = pd.DataFrame({'images': testing_image, 'captions': testing_caption})

model.to(device)
# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = 50265  # 30522

# set beam search parameters
model.config.eos_token_id = tokenizer.sep_token_id
model.config.max_length = 20
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

# find the maximun length for decoder model (in caption text)
max_length_train = np.max(train_df['captions'].apply(lambda x: len(x.split(' '))))
max_length_test = np.max(test_df['captions'].apply(lambda x: len(x.split(' '))))

max_length = max([max_length_train, max_length_test])


class IAMDataset(Dataset):
    def __init__(self, df, tokenizer, feature_extractor, decoder_max_length=max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.decoder_max_length = decoder_max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        img_path = self.df['images'][idx]
        caption = self.df['captions'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.tokenizer(caption, truncation=True,
                                padding="max_length",
                                max_length=self.decoder_max_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding


train_dataset = IAMDataset(train_df,
                           tokenizer=tokenizer,
                           feature_extractor=feature_extractor)
eval_dataset = IAMDataset(test_df,
                          tokenizer=tokenizer,
                          feature_extractor=feature_extractor)


# def compute_metrics(self, eval_pred):
#     pred_ids = eval_pred.predictions[0].argmax(-1)
#     label_ids = eval_pred.label_ids
#     tokenizer = self.tokenizer
#     pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
#     label_text = tokenizer.decode(label_ids[0], skip_special_tokens=True)
#     return {"accuracy": accuracy_score(label_ids[0], pred_ids), "f1": f1_score(label_ids[0], pred_ids, average='macro')}


captioning_model = 'VIT_Captioning'

training_args = TrainingArguments(
    output_dir=captioning_model,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    # predict_with_generate=True,
    # fp16=True,
    evaluation_strategy="epoch",
    do_train=True,
    do_eval=True,
    logging_steps=1024,
    save_steps=2048,
    warmup_steps=1024,
    #max_steps=1500, # delete for full training
    num_train_epochs=5, #TRAIN_EPOCHS
    overwrite_output_dir=True,
    save_total_limit=1,
)

# # instantiate trainer
trainer = Trainer(
    tokenizer=feature_extractor,
    model=model,
    args=training_args,
    # compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)
# Fine-tune the model, training and evaluating on the train dataset ----------------------------------------------------
trainer.train()
trainer.save_model('/opt/project/tmp/Image_Cationing_VIT_normal')
