import streamlit as st 
from time import sleep
from stqdm import stqdm  
import pandas as pd
from transformers import pipeline
import json
import spacy
import spacy_streamlit
from sentence_transformers import SentenceTransformer, util


def draw_all(
    key,
    plot=False,
):
    st.write(
        """
        # NLP Web App
        
        This Natural Language Processing Based Web App can do anything u can imagine with Text. 😱 
        
        This App is built using pretrained transformers which are capable of doing wonders with the Textual data.
        
        ```python
        # Key Features of this App.
        1. Advanced Text Summarizer
        2. Named Entity Recognition
        3. Sentiment Analysis
        4. Question Answering
        5. Text Completion
        6. Language Translation
        7. Text Similarity
        
        ```
        """
    )

with st.sidebar:
    draw_all("sidebar")


def main():
    st.title("NLP Web App")
    menu = ["--Select--","Summarizer","Named Entity Recognition",
            "Sentiment Analysis","Question Answering","Text Completion",
            "Language Translation", "Text Similarity"]
    choice = st.sidebar.selectbox("Choose What u wanna do !!", menu)


    if choice=="--Select--":
        
        st.write("""
                 
                 This is a Natural Language Processing Based Web App that can do   
                 anything u can imagine with the Text.
        """)
        
        st.write("""
                 
                Natural Language Processing (NLP) is a computational technique
                to understand the human language in the way they spoke and write.
        """)
        
        st.write("""
                 
                 NLP is a sub field of Artificial Intelligence (AI) to understand
                 the context of text just like humans.
        """)
        
        st.image('rob.webp')

    elif choice == "Summarizer":
        st.subheader("Text Summarization")
        st.write("Enter the Text you want to summarize!")
        raw_text = st.text_area("Your Text", "Enter Your Text Here")
        num_words = st.number_input("Enter Number of Words in Summary", min_value=10, max_value=100, value=50)
    
        if raw_text.strip() and num_words:
            summarizer = pipeline('summarization', model='facebook/bart-base')  # Use a better model
            input_length = len(raw_text.split())
            max_length = min(num_words, input_length - 1)  # Ensure max_length is less than input length
            min_length = max(20, max_length // 2)  # Set a reasonable min_length
    
            try:
                summary = summarizer(raw_text, min_length=min_length, max_length=max_length)
                result_summary = summary[0]['summary_text']
                st.write(f"Here's your Summary: {result_summary}")
            except ValueError as e:
                st.error(f"Error: {e}")


    elif choice=="Named Entity Recognition":
        nlp = spacy.load("en_core_web_trf")
        st.subheader("Text Based Named Entity Recognition")
        st.write(" Enter the Text below To extract Named Entities !")

        raw_text = st.text_area("Your Text","Enter Text Here")
        if raw_text !="Enter Text Here":
            doc = nlp(raw_text)
            for _ in stqdm(range(50), desc="Please wait a bit. The model is fetching the results !!"):
                sleep(0.1)
            spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe("ner").labels, title= "List of Entities")

    elif choice=="Sentiment Analysis":
        st.subheader("Sentiment Analysis")
        sentiment_analysis = pipeline("sentiment-analysis")
        st.write(" Enter the Text below To find out its Sentiment !")

        raw_text = st.text_area("Your Text","Enter Text Here")
        if raw_text !="Enter Text Here":
            result = sentiment_analysis(raw_text)[0]
            sentiment = result['label']
            for _ in stqdm(range(50), desc="Please wait a bit. The model is fetching the results !!"):
                sleep(0.1)
            if sentiment =="POSITIVE":
                st.write("""# This text has a Positive Sentiment.  🤗""")
            elif sentiment =="NEGATIVE":
                st.write("""# This text has a Negative Sentiment. 😤""")
            elif sentiment =="NEUTRAL":
                st.write("""# This text seems Neutral ... 😐""")

    elif choice=="Question Answering":
        st.subheader("Question Answering")
        st.write(" Enter the Context and ask the Question to find out the Answer !")
        question_answering = pipeline("question-answering")

        context = st.text_area("Context","Enter the Context Here")
        question = st.text_area("Your Question","Enter your Question Here")
        
        if context !="Enter Text Here" and question!="Enter your Question Here":
            result = question_answering(question=question, context=context)
            s1 = json.dumps(result)
            d2 = json.loads(s1)
            generated_text = d2['answer']
            generated_text = '. '.join(list(map(lambda x: x.strip().capitalize(), generated_text.split('.'))))
            st.write(f" Here's your Answer :\n {generated_text}")

    elif choice=="Text Completion":
        st.subheader("Text Completion")
        st.write(" Enter the uncomplete Text to complete it automatically using AI !")
        text_generation = pipeline("text-generation")
        message = st.text_area("Your Text","Enter the Text to complete")
        
        if message !="Enter the Text to complete":
            generator = text_generation(message)
            s1 = json.dumps(generator[0])
            d2 = json.loads(s1)
            generated_text = d2['generated_text']
            generated_text = '. '.join(list(map(lambda x: x.strip().capitalize(), generated_text.split('.'))))
            st.write(f" Here's your Generate Text :\n   {generated_text}")

    elif choice == "Language Translation":
        st.subheader("Language Translation")
        st.write(" Translate English text to French (Demo)")

        raw_text = st.text_area("Enter English Text", "Enter text here...")
        if raw_text and raw_text != "Enter text here...":
            translator = pipeline("translation_en_to_fr")
            result = translator(raw_text, max_length=100)
            st.write("### Translated Text:")
            st.success(result[0]['translation_text'])

    elif choice == "Text Similarity":
        st.subheader("Text Similarity Checker")
        st.write(" Compare two pieces of text to find how similar they are.")

        sent1 = st.text_area("Text 1", "Enter first sentence here...")
        sent2 = st.text_area("Text 2", "Enter second sentence here...")

        if sent1.strip() != "" and sent2.strip() != "":
            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            emb1 = model.encode(sent1, convert_to_tensor=True)
            emb2 = model.encode(sent2, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(emb1, emb2)
            st.write(f"### Similarity Score: {similarity.item():.4f}")

def trim_last(sent):
    if "." not in sent[-1]:
        return ''.join(sent)
    else:
        return ''.join(sent)

if __name__ == '__main__':
     main()
