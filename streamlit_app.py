import streamlit as st

from common import load_data, save_data, load_model, load_corpus_embedding, append_new_corpus, load_tokenized_data, search, download_files, ranking, train

@st.cache_resource()
def download_data():
    download_files()

@st.cache_resource()
def load_input_data():
    return load_data()

@st.cache_resource()
def load_model_and_corpus(passages, model_names):
    model_mapping = {}
    for model_name in model_names:
        model = load_model(model_name=model_name)
        corpus = load_corpus_embedding(model, passages, model_name=model_name)
        model_mapping[model_name] = {
            "model": model,
            "corpus": corpus
        }
    return model_mapping


def run(model_names, model_mapping, top_k=10):
    st.title('Demo Q&A')
    rankers = model_names[:]
    # rankers.append("ensemble")
    ranker = st.sidebar.radio('Loại mô hình ngôn ngữ', rankers, index=0)
    
    st.markdown("Bạn có thể nhập câu hỏi và câu trả lời để training model. " 
        + "Sau đó test hiệu quả của model bằng cách nhập các câu hỏi. "
        + "Bằng việc thu thập feedback (like/dislike) các câu trả lời để re-train lại model nếu cần.")
    st.text('')

    question = st.text_area('Nhập câu hỏi để train model')
    answers = []
    ans1 = st.text_area('Câu trả lời 1:')
    answers.append(ans1)

    ans2 = st.text_area('Câu trả lời 2: (không bắt buộc)')
    answers.append(ans2)

    new_passages = []
    dataset = []
    if len(ans1) > 0:
        dataset.append([question, answers[0], 0.99])
        new_passages.append([question, answers[0]])
    
    if len(ans2) > 0:
        dataset.append([question, answers[1], 0.97])
        new_passages.append([question, answers[1]])

    if st.button('Training'):
        print("Trigger training model ", ranker)
        with st.spinner('Training ......'):
            if ranker in model_names:
                model = model_mapping[ranker]["model"]

                # train model on new data
                model = train(model, ranker, dataset)

                # append new passages into the existing passages
                passages.extend(new_passages)
                save_data(passages)

                # encode corpus embedding for the whole passages
                new_corpus = append_new_corpus(model_mapping[ranker]["corpus"], model, new_passages, ranker)
                
                # update new model & corpus for this model_name
                model_mapping[ranker]["corpus"] = new_corpus
                model_mapping[ranker]["model"] = model

    input_text = []
    comment = st.text_area('Bạn có thể test thử model bằng cách nhập câu hỏi vào ô bên dưới')
    input_text.append(comment)

    is_ranking = st.checkbox("Ranking lại kết quả tìm kiếm")

    if st.button('Tìm kiếm'):
        with st.spinner('Searching ......'):
            if input_text != '':
                print(f'Input: ', input_text)
                query = input_text[0]
                if ranker == "ensemble":

                    ids = []
                    for model_name in model_names:
                        model = model_mapping[model_name]["model"]
                        corpus = model_mapping[model_name]["corpus"]
                        results, hits = search(model, corpus, query, passages, top_k=top_k)
                        
                        for hit in hits:
                            ids.append(hit["corpus_id"])
                    
                    ids = set(ids)
                    hits = [{"corpus_id": id} for id in ids]
                    results = [passages[i] for i in ids]

                    if is_ranking:
                        results, _ = ranking(hits, query, passages, top_k=top_k)
                else:
                    print("Search answers with model ", ranker)
                    model = model_mapping[ranker]["model"]
                    corpus = model_mapping[ranker]["corpus"]
                    results, hits = search(model, corpus, query, passages, top_k=top_k)
                    
                    if is_ranking:
                        results, _ = ranking(hits, query, passages, top_k=top_k)

                for result in results:
                    st.success(f"{str(result)}")


if __name__ == '__main__':
    model_names = [
        "keepitreal/vietnamese-sbert",
        # "sentence-transformers/all-MiniLM-L12-v2",
        # "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        # "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ]

    download_data()
    passages = load_input_data()
    model_mapping = load_model_and_corpus(passages, model_names)
    run(model_names, model_mapping, top_k=10)