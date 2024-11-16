import pandas as pd
import re
import requests
from langchain_community.document_loaders import BSHTMLLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import time
import streamlit as st

# Define the model to store the extracted professor information
class ProfessorInfo(BaseModel):
    """Information about a professor."""
    email: Optional[str] = Field(description="The email address of the professor.")
    research_field: Optional[str] = Field(description="The research field of the professor.")
    research_interests: Optional[str] = Field(description="The research interests of the professor.")
    department: Optional[str] = Field(description="The department of the professor.")

class ExtractionData(BaseModel):
    """Extracted information about a professor."""
    professor_info: List[ProfessorInfo] = Field(description="The list of extracted professor information.")

# Set up the output parser
parser = PydanticOutputParser(pydantic_object=ExtractionData)

# Streamlit app
st.title("Professor Information Extractor")

# API Key Input
api_key = st.text_input("Enter your Gemini API Key", type="password")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file containing URLs", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file containing the URLs
    df = pd.read_csv(uploaded_file, encoding='utf-8')

    # Display the first few rows of the DataFrame
    st.write("Uploaded DataFrame:")
    st.dataframe(df.head())

    # Column selection for URLs with default to "Select URL Column"
    url_column = st.selectbox("Select the column containing URLs", ["Select URL Column"] + list(df.columns), index=0)

    # Button to start processing
    if st.button("Start Processing"):
        if api_key and url_column != "Select URL Column":
            # Define the chat model with the provided API key
            model = ChatGoogleGenerativeAI(api_key=api_key, model="gemini-1.5-flash")

            # Define a custom prompt to provide instructions and any additional context.
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an expert at identifying key information about professors. "
                        "Only extract information if it is explicitly provided. If no information can be found, extract nothing. "
                        "Wrap the output in `json` tags\n{format_instructions}",
                    ),
                    (
                        "human",
                        "Extract the professor's email, research field, research interests, and department from the following text: {text}"
                    ),
                ]
            ).partial(format_instructions=parser.get_format_instructions())

            # Chain to process text
            chain = prompt | model | parser

            # Function to process a single URL and extract information
            def process_url(url):
                modified_url = "https://r.jina.ai/" + url

                # Download the content from the modified URL
                response = requests.get(modified_url)

                # Write the content to an HTML file
                html_file = "temp.html"
                with open(html_file, "w", encoding="utf-8") as f:
                    f.write(response.text)

                # Load the HTML content using BSHTMLLoader
                loader = BSHTMLLoader(html_file)
                document = loader.load()[0]

                # Clean up the page content
                document.page_content = re.sub("\n\n+", "\n", document.page_content)
                document.page_content = re.sub(r'\s+', ' ', document.page_content).strip()
                document.page_content = re.sub(r'\.{2,}', '.', document.page_content)

                # Extract the professor's information
                result = chain.invoke({"text": document.page_content})

                return result

            # Ensure the 'name', 'email', and 'research' columns are of string type (object)
            required_columns = ['email', 'research_field', 'research_interests', 'department']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = None  # Add the column with None values if it doesn't exist

            # Initialize an empty list to store results
            extracted_data = []

            # Loop through each URL in the selected column
            for url in df[url_column]:
                try:
                    st.write(f"Processing URL: {url}")
                    result = process_url(url)
                    st.write(result)

                    # Extract the first professor info (assuming there's at least one)
                    if result.professor_info:
                        professor = result.professor_info[0]  # Accessing the first item in the list
                        # Append the extracted data to the dataframe
                        df.loc[df[url_column] == url, 'email'] = professor.email
                        df.loc[df[url_column] == url, 'research_field'] = professor.research_field
                        df.loc[df[url_column] == url, 'research_interests'] = professor.research_interests
                        df.loc[df[url_column] == url, 'department'] = professor.department
                        st.write(f"Extracted data for {url}: {professor}")

                    # Wait for 5 seconds to avoid overloading the server
                    time.sleep(5)
                except Exception as e:
                    st.write(f"Error processing {url}: {e}")

            # Display the updated DataFrame
            st.write("Updated DataFrame:")
            st.dataframe(df)

            # Save the updated DataFrame back to a CSV file
            csv = df.to_csv(index=False, encoding='utf-8')
            st.download_button(
                label="Download updated CSV",
                data=csv,
                file_name='updated_professors.csv',
                mime='text/csv',
            )

            st.write("Data extraction and saving completed!")
        elif not api_key and url_column == "Select URL Column":
            st.warning("Please enter your Gemini API Key and select the column containing URLs to proceed.")
        elif not api_key:
            st.warning("Please enter your Gemini API Key to proceed.")
        elif url_column == "Select URL Column":
            st.warning("Please select the column containing URLs to proceed.")
else:
    st.warning("Please enter your Gemini API Key and upload a CSV file to proceed.")