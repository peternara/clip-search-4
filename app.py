import streamlit as st
import clip
import torch
from torch.nn import functional as F
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
@st.cache
def get_clip():
    model, preprocess = clip.load("ViT-B/32", device=device)

    return model, preprocess

@st.cache
def get_dataset():
    filenames = json.loads(open("index/filenames.json").read())
    x = torch.load("index/clip_codes.pt").to(device)
    if device == "cpu":
        x = x.float()

    return x, filenames

st.title('CLIP search')

query = st.text_input("Search")

model, _ = get_clip()
if query:
    text = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    text_features = F.normalize(text_features, dim=-1)
    dataset, filenames = get_dataset()
    similarity = F.cosine_similarity(text_features, dataset)
    res, idx = similarity.topk(20, dim=0)
    
    urls = [f"https://s3.amazonaws.com/open-images-dataset/validation/{filenames[i]}" for i in idx]
    labels = [f"{int(x*100)}" for x in res]
    st.image(urls, labels, width=200)