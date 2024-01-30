import unittest
from unittest.mock import patch, MagicMock
from vectordb import create_findex

class TestCreateFindex(unittest.TestCase):
    @patch('langchain_community.vectorstores.FAISS')
    @patch('langchain_openai.OpenAIEmbeddings')
    @patch('streamlit.secrets')
    def test_create_findex(self, mock_secrets, mock_embeddings, mock_faiss):
        """
        Test the create_findex function by mocking secrets, embeddings, and the FAISS index.
        """
        # Mock the secrets
        mock_secrets.return_value = {'OPENAI_API_KEY': 'test_key'}

        # Mock the embeddings
        mock_embeddings.return_value = MagicMock()

        # Mock the FAISS index
        mock_faiss.from_documents.return_value = MagicMock()

        # Call the function with test data
        create_findex('test_session', ['test_chunk'], mock_embeddings)

        # Assert that the function calls were made correctly
        mock_secrets.assert_called_once()
        mock_embeddings.assert_called_once_with('test_key', 'text-embedding-3-small')
        mock_faiss.from_documents.assert_called_once_with(['test_chunk'], mock_embeddings)

if __name__ == '__main__':
    unittest.main()