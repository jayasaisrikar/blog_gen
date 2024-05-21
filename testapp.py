import base64
import re
import string
import urllib.request

import nltk
import streamlit as st
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate

# Download nltk resources
#nltk.download("punkt")

# Download the background image
background_image_url = "https://res.cloudinary.com/practicaldev/image/fetch/s--Rpm5i2vq--/c_limit%2Cf_auto%2Cfl_progressive%2Cq_66%2Cw_880/https://dev-to-uploads.s3.amazonaws.com/uploads/articles/zzszorn7hxyvther2lea.gif"
background_image_path = "background_image.gif"
urllib.request.urlretrieve(background_image_url, background_image_path)

# Function to get the response from the language model


def getresponse(input_text, no_words, blog_style, llm):
    template = """
    Create an insightful and engaging blog post on the topic "{input_text}" for the {blog_style} audience. Share your expertise and opinions while providing valuable information. Craft a narrative that captivates the readers and keeps them interested throughout the article. Aim to cover the key aspects within a limit of {no_words} words. Remember to include examples, real-world scenarios, and any recent developments related to the topic. Your goal is to deliver a well-rounded and informative piece that resonates with the {blog_style} community.
    """

    prompt = PromptTemplate(input_variables=["blog_style", 'input_text', 'no_words'],
                            template=template)
    response = llm(prompt.format(blog_style=blog_style,
                                 input_text=input_text, no_words=no_words))

    # Post-process to meet word count more accurately
    response = re.sub(f"[{string.punctuation}]", "", response)
    words = response.split()
    response = ' '.join(words[:int(no_words)])

    return response, len(words)


# Set page configuration
st.set_page_config(
    page_title="Blog Generator App",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state='collapsed'
)


st.markdown(
    f"""
    <style>
    body {{
        background-image: url('data:image/gif;base64,{base64.b64encode(open(background_image_path, "rb").read()).decode()}');
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# Add a header with instructions
st.markdown("""
# Blog Generator App
Welcome to the Blog Generator App! Enter the topic, choose the audience, and specify the word count to generate a blog instantly.
""")

# Sidebar for user input
st.sidebar.markdown("## Enter details below")
input_text = st.sidebar.text_input("Enter the blog topic:")
no_words = st.sidebar.text_input('No of Words')
blog_style = st.sidebar.selectbox('Writing the blog for', [
    'Researchers', 'Data Scientist', 'Common People', 'Professionals'])

# Load the language model


@st.cache(allow_output_mutation=True)
def load_language_model():
    return CTransformers(model="models\llama-2-7b-chat.ggmlv3.q8_0.bin",
                         model_type='llama',
                         config={'max_new_tokens': 256,
                                 'temperature': 0.01})


llm = load_language_model()

# Generate and display the response
if st.sidebar.button('Generate', key="generate_button", help="Click to generate the blog"):
    if not input_text or not no_words:
        st.sidebar.warning(
            "Please enter both the blog topic and the desired word count.")
    else:
        try:
            with st.spinner("Generating..."):
                # Generate response using the getresponse function
                response, word_count = getresponse(
                    input_text, no_words, blog_style, llm)
                st.sidebar.success("Blog generated successfully!")

                # Display entire generated content using st.text_area
                st.text_area("Generated Blog", response, height=800)

                # Display the actual word count
                st.sidebar.markdown(f"Actual Word Count: {word_count}")

        except Exception as e:
            st.sidebar.error(f"An error occurred: {str(e)}")

# Add a footer
st.sidebar.markdown("<div style='text-align: center;'><p>Blog Generator by Jaya Sai Srikar</p></div>",
                    unsafe_allow_html=True)
