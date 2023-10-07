import streamlit as st
import hashlib

from common import load_data, save_data, load_model, load_corpus_embedding, append_new_corpus, load_tokenized_data, search, download_files, ranking, train

if 'statistic' not in st.session_state:
    st.session_state.statistic = None

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

@st.cache_resource()
def get_md5(input):
    md5_hash = hashlib.md5()
    md5_hash.update(input.encode('utf-8'))
    return md5_hash.hexdigest()


def run(model_names, model_mapping, top_k=10):
    st.title('Demo Q&A')
    rankers = model_names[:]
    ranker = st.sidebar.radio('Loại mô hình ngôn ngữ', rankers, index=0)
    
    st.markdown("Bạn có thể nhập câu hỏi và câu trả lời để training model. " 
        + "Sau đó test hiệu quả của model bằng cách nhập các câu hỏi. "
        + "Bằng việc thu thập feedback (like/dislike) các câu trả lời để re-train lại model nếu cần.")
    st.text('')

    question = st.text_area('Nhập câu hỏi để train model')
    answers = []
    ans1 = st.text_area('Câu trả lời 1:')
    if len(ans1.strip()) > 0:
        answers.append(ans1.strip())

    ans2 = st.text_area('Câu trả lời 2: (không bắt buộc)')
    if len(ans2.strip()) > 0:
        answers.append(ans2.strip())

    new_passages = []
    dataset = []
    if len(answers) > 0:
        dataset.append([question, answers[0], 0.99])
        new_passages.append([question, answers[0]])
    
    if len(answers) > 1:
        dataset.append([question, answers[1], 0.97])
        new_passages.append([question, answers[1]])
    
    print("dataset: ", dataset)

    if question != "" or len(answers) > 0:
        if st.button('Training'):
            print("->", question, answers)
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

    query = st.text_area('Bạn có thể test thử model bằng cách nhập câu hỏi vào ô bên dưới')
    query_md5 = get_md5(query)

    hits = []
    if st.button('Tìm kiếm'):
        with st.spinner('Searching ......'):
            if query != '':
                print(f'Input: ', query)
                print("Search answers with model ", ranker)
                model = model_mapping[ranker]["model"]
                corpus = model_mapping[ranker]["corpus"]
                results, hits = search(model, corpus, query, passages, top_k=top_k)
    
                print(hits)

                # update statistic
                st.session_state.statistic = {
                    query_md5: {
                        "query": query,
                        "hits": hits,
                        "status": None
                    }
                }
                            
                # collect results ids
                ids = [hit['corpus_id'] for hit in hits]

                # Display items and like/dislike buttons
                for id in ids:
                    st.success(f"{str(passages[id])}")
                    
                    # # Create columns for like and dislike buttons
                    # col1, col2 = st.columns(2)
                    
                    # with col1:
                    #     # Add a like button
                    #     if col1.button("Like", key=f"like-{id}"):
                    #         if st.session_state.statistic[query_md5]["status"][id] is None:
                    #             st.session_state.statistic[query_md5]["status"] = {
                    #                 id: {
                    #                     "like": 1
                    #                 }
                    #             }
                    #         else:
                    #             st.session_state.statistic[query_md5]["status"][id]["like"] += 1
                        
                    # with col2:
                    #     # Add a dislike button
                    #     if col2.button("Dislike", key=f"dislike-{id}"):
                    #         if st.session_state.statistic[query_md5]["status"][id] is None:
                    #             st.session_state.statistic[query_md5]["status"] = {
                    #                 id: {
                    #                     "like": -1
                    #                 }
                    #             }
                    #         else:
                    #             st.session_state.statistic[query_md5]["status"][id]["like"] -= 1

                # print(st.session_state.statistic)      

if __name__ == '__main__':
    model_names = [
        # "keepitreal/vietnamese-sbert",
        "sentence-transformers/all-MiniLM-L12-v2",
        # "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        # "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ]

    download_data()
    passages = load_input_data()
    model_mapping = load_model_and_corpus(passages, model_names)
    run(model_names, model_mapping, top_k=10)