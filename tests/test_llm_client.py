# Copyright 2026 Justin Cook
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Pre-mock modules to avoid ImportErrors if not installed
sys.modules['google'] = MagicMock()
sys.modules['google.genai'] = MagicMock()
sys.modules['google'].genai = sys.modules['google.genai']
sys.modules['openai'] = MagicMock()
sys.modules['vertexai'] = MagicMock()

from cv_maker import llm_client
from cv_maker.models import JobDescription

class TestLLMClient(unittest.TestCase):
    def setUp(self):
        # Mock cache file to avoid FS errors
        with patch.object(llm_client.LLMClient, '_load_cache', return_value=[]):
             self.client = llm_client.LLMClient(provider="gemini")
        # We need to ensure cache is mocked during tests too if called
        self.mock_cache_patch = patch.object(llm_client.LLMClient, '_load_cache', return_value=[])
        self.mock_cache = self.mock_cache_patch.start()

    def tearDown(self):
        self.mock_cache_patch.stop()

    def test_call_llm_gemini(self):
        # Setup mocks
        mock_genai = sys.modules['google.genai']
        
        # Create explicit chain
        mock_client = MagicMock()
        mock_models = MagicMock()
        mock_generate_content = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Gemini Response"
        
        # Link them
        mock_genai.Client.return_value = mock_client
        # Simplest approach: just assign the attribute
        mock_client.models = mock_models
        # Link generate_content
        mock_models.generate_content = mock_generate_content
        mock_generate_content.return_value = mock_response

        # Use patch.dict for env vars
        with patch.dict(os.environ, {"GEMINI_API_KEY": "mock_key"}, clear=True):
            # Need to also patch discover_models since it's called
            with patch.object(llm_client.LLMClient, 'discover_models', return_value=['gemini-discovery']):
                client = llm_client.LLMClient(provider="gemini")
                result = client._call_llm("Test Prompt")
                
                self.assertEqual(result, "Gemini Response")

    def test_call_llm_openai(self):
        mock_openai = sys.modules['openai']
        
        # Explicit chain
        mock_client = MagicMock()
        mock_chat = MagicMock()
        mock_completions = MagicMock()
        mock_create = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        
        mock_message.content = "OpenAI Response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat = mock_chat
        mock_chat.completions = mock_completions
        mock_completions.create = mock_create
        mock_create.return_value = mock_response

        # Use patch.dict to set OPENAI 
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-mock-openai-key"}, clear=True):
            client = llm_client.LLMClient(provider="openai")
            result = client._call_llm("Test Prompt")

            self.assertEqual(result, "OpenAI Response")

    def test_tailor_cv(self):
        client = llm_client.LLMClient()
        fake_json = """
        {
            "name": "Jane",
            "title": "Dev",
            "contact_info": "Contact",
            "executive_summary": "Sum",
            "competencies": [["Cat", "Skill"]],
            "experience": [],
            "earlier_experience": [{"title": "Old Role", "company": "Old Corp", "summary": "Did stuff"}],
            "projects": [],
            "education": [],
            "certifications": ""
        }
        """
        with patch.object(client, '_call_llm', return_value=fake_json):
            jd = JobDescription(raw_text="JD", role_title="Dev", key_skills=["Python"], summary="Role")
            cv_data = client.tailor_cv("Master CV", jd)
            self.assertEqual(cv_data.name, "Jane")
            self.assertEqual(len(cv_data.earlier_experience), 1)
            self.assertEqual(cv_data.earlier_experience[0].company, "Old Corp")

    def test_tailor_cv_summary_prompt(self):
        client = llm_client.LLMClient()
        fake_json = '{}' # invalid json but we just want to check input prompt
        
        with patch.object(client, '_call_llm', return_value=fake_json):
            jd = JobDescription(raw_text="JD", role_title="Dev", key_skills=["Python"], summary="Role")
            
            # Case 1: summarize_years=10 (Default)
            client.tailor_cv("CV", jd, summarize_years=10)
            call_args = client._call_llm.call_args[0][0]
            self.assertIn("Identify ALL roles ending in", call_args)
            self.assertIn("CRITICAL: Any role that ended BEFORE", call_args)
            
            # Case 2: summarize_years=0 (Disabled)
            client.tailor_cv("CV", jd, summarize_years=0)
            call_args = client._call_llm.call_args[0][0]
            self.assertIn("Select the top most relevant roles", call_args)
            self.assertIn("Do NOT use the 'earlier_experience' array", call_args)

    def test_tailor_cv_robust_unpacking(self):
        """Test that single-element lists in JSON don't crash the unpacked tuples."""
        client = llm_client.LLMClient()
        fake_json = """
        {
            "name": "Test",
            "experience": [
                {
                    "title": "Role",
                    "company": "Co",
                    "bullets": [["Bullet 1"], ["Bullet 2", "Desc 2"], "StringBullet"]
                }
            ],
            "competencies": [["Comp 1"]],
            "projects": [["Proj 1"]]
        }
        """
        with patch.object(client, '_call_llm', return_value=fake_json):
            cv = client.tailor_cv("Master", JobDescription("JD"))
            
            # Bullets
            bullets = cv.experience[0].bullets
            self.assertEqual(len(bullets), 3)
            self.assertEqual(bullets[0], ("Bullet 1", "")) # Padded
            self.assertEqual(bullets[1], ("Bullet 2", "Desc 2")) # Normal
            self.assertEqual(bullets[2], ("StringBullet", "")) # String case
            
            # Competencies
            self.assertEqual(cv.competencies[0], ("Comp 1", ""))
            
            # Projects
            self.assertEqual(cv.projects[0], ("Proj 1", ""))

if __name__ == '__main__':
    unittest.main()
