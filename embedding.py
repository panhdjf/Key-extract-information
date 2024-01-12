import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS



def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask
  
model_name_or_path = "VietnamAIHub/Vietnamese_llama2_7B_8K_SFT_General_domain"

cache_dir = "/home/vinbig/Documents/PA_Modeling/Retrieval_pdf/tmp"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True,cache_dir=cache_dir)

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    # load_in_8bit=True,
    torch_dtype=torch.float32,
    pretraining_tp=1,
    # use_auth_token=True,
    # trust_remote_code=True,
    cache_dir=cache_dir,
)


sentences = ['This is an example sentence', 'Each sentence is converted']

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    model_output = model(**encoded_input)
# model_output = model(**encoded_input)
import torch
import torch.nn.functional as F


sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

# print(sentence_embeddings[0].shape)

# vectorstore = FAISS.from_texts(sentences, embedding=mean_pooling)

# from datasets import load_dataset

# squad = load_dataset("squad", split="validation").shuffle(seed=42).select(range(100))


def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v for k, v in encoded_input.items()}
    with torch.no_grad():
        model_output = model(**encoded_input)
    return mean_pooling(model_output, encoded_input["attention_mask"])


squad_with_embeddings = sentences.map(
    lambda x: {"embeddings": get_embeddings(sentences)}
)

squad_with_embeddings.add_faiss_index(column="embeddings")

# question = "Who headlined the halftime show for Super Bowl 50?"
# question_embedding = get_embeddings([question]).cpu().detach().numpy()

# scores, samples = squad_with_embeddings.get_nearest_examples(
#     "embeddings", question_embedding, k=3
# )
# question = "How can I load a dataset offline?"
# question_embedding = get_embeddings([question]).numpy()
# print(question_embedding.shape)
print(squad_with_embeddings)