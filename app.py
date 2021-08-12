import streamlit as st
import awesome_streamlit as ast
import scripts.pages.recorded_audio
import scripts.pages.record_audio
import scripts.pages.home
import scripts.pages.model_summary

# create the pages
PAGES = {
    "Home" : scripts.pages.home,
    "Model Summary" : scripts.pages.model_summary,
    "Choose Audio": scripts.pages.recorded_audio,
    "Record your own voice": scripts.pages.record_audio,
    
}


# render the pages
def main():
    
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    
    page = PAGES[selection]
    with st.spinner(f"Loading {selection} ..."):
        ast.shared.components.write_page(page)
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app is an end-to-end solution that is capable of transcribing a speech to text in the Amharic language.
        """
                    )  

if __name__ == "__main__":
    main()