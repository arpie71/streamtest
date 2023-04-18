mport streamlit as st
import spacy
import spacy_streamlit
from spacy_streamlit import visualize_ner
from spacy_streamlit import process_text
from spacy_streamlit import load_model

models = ["en_core_web_lg", "NERpipeline/models/combinednlp"]
#inp_txt = st.text_input("Enter your text")
st.title("History Lab NER model")
default_text = "Sundar Pichai is the CEO of Google."
default_text = 'Henry Alfred Kissinger (born Heinz Alfred Kissinger, May 27, 1923) is a German-born American politician, diplomat, and geopolitical consultant who served as United States Secretary of State and National Security Advisor under the presidential administrations of Richard Nixon and Gerald Ford. A Jewish refugee who fled Nazi Germany with his family in 1938, he became National Security Advisor in 1969 and U.S. Secretary of State in 1973. For his actions negotiating a ceasefire in Vietnam, Kissinger received the 1973 Nobel Peace Prize under controversial circumstances, with two members of the committee resigning in protest.'
spacy_model = st.sidebar.selectbox("Model name", [ "en_core_web_sm"])
text = st.sidebar.text_area("Text to analyze", "This is a text")
doc = process_text(spacy_model, text)
#spacy_streamlit.visualize(models, inp_txt)
#nlp = spacy.load('NERpipeline/models/combinednlp')
#doc = nlp(default_text)
nlp = load_model(spacy_model)
visualize_ner(doc, labels=nlp.get_pipe("ner").labels, attrs=["text", "start_char", "end_char", "label_", "kb_id_"])
s = nlp(doc)
jsonout = []
for ent in s.ents:
        jsonout.append([ent.text, ent.start_char, ent.end_char, ent.label_, ent.kb_id_])

    import pandas as pd
    df = pd.DataFrame(jsonout, columns = ['Entity','Start','End','NERlab','Wikidata'])
    #json_string = json.dumps(jsonout)

    #st.json(json_string, expanded=True)

    #st.download_button(
    #    label="Download JSON",
    #    file_name="data.json",
    #    mime="application/txt",
    #    data=jsonout,
    #)

    @st.experimental_memo
    def convert_df(df):
           return df.to_csv(index=False).encode('utf-8')


        csv = convert_df(df)

        st.sidebar.download_button(
                   "Press to Download",
                   csv,
                   "file.csv",
                   "text/csv",
                   key='download-csv'
                )
