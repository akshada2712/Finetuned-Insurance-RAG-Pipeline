from dotenv import load_dotenv
import streamlit as st 
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma 
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os 
import uuid 


class InsuranceBot:
    def __init__(self):
        load_dotenv()

        self.store = {}

        self.llm = ChatOpenAI(
            model_name = "ft:gpt-4o-mini-2024-07-18:personal::B1zvbetx",
            openai_api_key = os.getenv('OPENAI_API_KEY'),
            temperature = 0.3  # Lower temperature for more factual responses
        )

        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            persist_directory="./niva_bupa_vectorstore",
            embedding_function=self.embeddings
        )

        # Increase k to retrieve more context and add search type
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 5},  
            search_type="mmr"  # Maximum Marginal Relevance to get diverse but relevant results
        )

        # Improved prompt with better context handling
        self.prompt = ChatPromptTemplate.from_template(
            "You are an expert on Niva Bupa insurance policies. Answer the user's query based ONLY on the given document context. "
            "If the document context doesn't contain relevant information to answer the question, politely explain that you don't have that "
            "specific information and offer to help with related topics you can address. "
            "If you've answered a similar question before, you can reference your previous answer but verify it against the current context. "
            "Don't make up information or policies that aren't in the provided context.\n\n"
            "Previous conversation context: {chat_history}\n\n"
            "Document context: {context}\n\n"
            "User Query: {input}\n\n"
            "Answer:"
        )
        
        # Create chains
        self.document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.retrieval_chain = create_retrieval_chain(self.retriever, self.document_chain)
    
    def generate_answer(self, query, chat_history):
        
        # Get the raw response
        response = self.retrieval_chain.invoke({
            "input": query,
            "chat_history": chat_history
        })
        
        # If no relevant documents found, provide a more controlled response
        if not response.get("context") or len(response.get("context", [])) == 0:
            return {
                "answer": "I don't have enough information in my knowledge base to answer this specific question accurately. "
                          "Could you please rephrase your question or ask about a different aspect of Niva Bupa insurance policies?",
                "context": []
            }
        
        return response

       
def initialize_session_state():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'conversation_ended' not in st.session_state:
        st.session_state.conversation_ended = False
    if 'context' not in st.session_state:
        st.session_state.context = {}  # Store retrieved context for reference

def main():

    st.title("Insurance Policy QA Bot")
    st.markdown("##### Ask questions about Niva Bupa insurance policies")

    initialize_session_state()

    qa_bot = InsuranceBot()
    
    # Move chat input up before displaying chat history
    prompt = None
    if not st.session_state.conversation_ended:
        prompt = st.chat_input("Ask your question about Niva Bupa insurance policies...")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.conversation_ended:
        if st.button("Start New Conversation"):
            st.session_state.conversation_ended = False
            st.session_state.chat_history = []
            st.session_state.context = {}
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
        return
    
    if prompt:
        # Check for exit commands
        exit_commands = ["exit", "quit", "thank you", "thanks", "bye"]
        if any(cmd in prompt.lower() for cmd in exit_commands):
            st.session_state.conversation_ended = True
            with st.chat_message("assistant"):
                st.write("Thank you for using our service! The conversation has ended. You can start a new one if needed.")
            st.rerun()
            return
        
        # Retrieve last N turns of conversation for context
        recent_history = st.session_state.chat_history[-6:] if len(st.session_state.chat_history) > 6 else st.session_state.chat_history
        
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message('user'):
            st.write(prompt)

        # Format recent history for context
        chat_history_text = '\n'.join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in recent_history
        ])

        with st.spinner("Thinking..."):
            response = qa_bot.generate_answer(prompt, chat_history_text)

        with st.chat_message("assistant"):
            st.write(response["answer"])

        st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})
        
        # Store the relevant documents for this question
        docs_and_scores = qa_bot.vectorstore.similarity_search_with_score(prompt, k=5)
        st.session_state.context[prompt] = docs_and_scores
        
        with st.expander("View source documents"):
            for i, (doc, score) in enumerate(docs_and_scores, 1):
                if score < 0.7:  # Only show reasonably relevant documents
                    st.markdown(f"**Document {i}** (Relevance: {score:.4f})")
                    st.markdown("---")
                    st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                    st.markdown("---")

if __name__ == "__main__":
    main()