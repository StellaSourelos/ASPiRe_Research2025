import unittest
import json
from proto13 import app

class RagPrototypeTestCase(unittest.TestCase):
    def setUp(self):
        # Set up Flask test client
        self.app = app.test_client()
        self.app.testing = True

    def test_homepage_get(self):
        # Test GET on homepage loads correctly
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Enter your question:', response.data)

    def test_homepage_post_empty_query(self):
        # Test POST with empty query returns error
        response = self.app.post('/', data={'query': ''})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Empty query provided', response.data)

    def test_homepage_post_query(self):
        # Test POST with sample query returns a generated answer
        response = self.app.post('/', data={'query': 'What is AI?'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Generated Answer:', response.data)

    def test_feedback_post(self):
        # Test posting feedback
        data = {
            'query': 'What is AI?',
            'answer': 'Artificial Intelligence is...',
            'docs': json.dumps([1,2]),
            'thumbs': 'up',
            'edit': 'No edit'
        }
        response = self.app.post('/feedback', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Thanks for the feedback!', response.data)

    def test_dashboard(self):
        # Test dashboard loads
        response = self.app.get('/dashboard')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Feedback Dashboard', response.data)

    def test_update_doc_invalid(self):
        # Test update document with missing data returns error
        response = self.app.post('/update_doc', data={})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Invalid data', response.data)

if __name__ == '__main__':
    unittest.main()

