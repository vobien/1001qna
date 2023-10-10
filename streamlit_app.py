import streamlit as st
import hashlib

from common import load_data, save_data, load_model, load_corpus_embedding, append_new_corpus, load_tokenized_data, search, download_files, ranking, train

# Initialize session state to store result data
if "results" not in st.session_state:
    st.session_state.results = {}


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

def get_md5(content):
    md5_hash = hashlib.md5()
    md5_hash.update(content.encode('utf-8'))
    return md5_hash.hexdigest()


class Result:
    def __init__(self, id, score, likes=0):
        self.id = id
        self.score = score
        self.likes = likes

# Function to render a result and handle likes/dislikes
def render_result(result):
    st.write(f"{passages[result.id][0]} - {passages[result.id][1]}")
    st.write(f"Likes: {result.likes}")

    col1, col2 = st.columns([0.1, 0.9])

    # Create buttons for liking and disliking
    like_btn = col1.button("Like", key=f"like-{result.id}")
    dislike_btn = col2.button("Dislike", key=f"dislike-{result.id}")
    
    if like_btn:
        likes = st.session_state.results[result.id].likes + 1
        st.session_state.results[result.id] = Result(result.id, result.score, likes)
        st.rerun()

    if dislike_btn:
        likes = st.session_state.results[result.id].likes - 1
        st.session_state.results[result.id] = Result(result.id, result.score, likes)
        st.rerun()

    
    
def run(model_names, model_mapping, passages, top_k=10):
    st.title('Demo Q&A')
    rankers = model_names[:]
    ranker = st.sidebar.radio('Loại mô hình ngôn ngữ', rankers, index=0)
    
    st.markdown("Bạn có thể nhập câu hỏi và câu trả lời để training model. " 
        + "Sau đó test hiệu quả của model bằng cách nhập các câu hỏi. "
        + "Bằng việc thu thập feedback (like/dislike) các câu trả lời để re-train lại model nếu cần.")
    st.text('')

    with st.form(key="train_form"):
        question = st.text_area('Nhập câu hỏi để train model')
        ans1 = st.text_area('Câu trả lời 1:')
        ans2 = st.text_area('Câu trả lời 2: (không bắt buộc)')
        train_button = st.form_submit_button(label="Training")

    # Check if the Training button was clicked
    if train_button:
        # collect answers
        answers = []
        if len(ans1.strip()) > 0:
            answers.append(ans1.strip())
        if len(ans2.strip()) > 0:
            answers.append(ans2.strip())

        if question.strip() != "" and len(answers) > 0:
            new_passages = []
            dataset = []
            if len(answers) > 0:
                dataset.append([question, answers[0], 0.99])
                new_passages.append([question, answers[0]])
            
            if len(answers) > 1:
                dataset.append([question, answers[1], 0.98])
                new_passages.append([question, answers[1]])
            
            print("Trigger training model ", ranker)
            with st.spinner('Training ......'):
                if ranker in model_names:
                    model = model_mapping[ranker]["model"]

                    # train model on new data
                    model = train(model, ranker, dataset, epochs=30)

                    # append new passages into the existing passages
                    passages.extend(new_passages)
                    save_data(passages)

                    # encode corpus embedding for the whole passages
                    new_corpus = append_new_corpus(model_mapping[ranker]["corpus"], model, new_passages, ranker)
                    
                    # update new model & corpus for this model_name
                    model_mapping[ranker]["corpus"] = new_corpus
                    model_mapping[ranker]["model"] = model

    with st.form(key="search_form"):
        query = st.text_area('Bạn có thể test thử model bằng cách nhập câu hỏi vào ô bên dưới')
        search_button = st.form_submit_button(label="Tìm kiếm")
    
    # Check if the Search button was clicked
    if search_button:
        hits = []
        print("Press Search button, query: ", query)
        del st.session_state["results"]

        if query != "":
            query_md5 = get_md5(query)
            with st.spinner('Searching ......'):
                print(f'Input: ', query)
                print("Search answers with model ", ranker)
                model = model_mapping[ranker]["model"]
                corpus = model_mapping[ranker]["corpus"]
                results, hits = search(model, corpus, query, passages, top_k=top_k)
                print(hits)

                st.session_state.results = {}
                for hit in hits:
                    id = hit["corpus_id"]
                    score = hit["score"]
                    st.session_state.results[id] = Result(id, score, likes=0)
                st.rerun()

    if query != "":
        # result area below Search button
        for id, result in st.session_state.results.items():
            render_result(result)

        # collect statistic of results
        if len(st.session_state.results) > 0:
            new_training_data = []
            for id, result in st.session_state.results.items():
                if result.likes != 0:
                    label = result.score + result.likes * 0.1
                    ans = f"{passages[result.id][0]},{passages[result.id][1]}"
                    new_training_data.append([query, ans, label])
            print("new training data: ", new_training_data)

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
    run(model_names, model_mapping, passages, top_k=10)