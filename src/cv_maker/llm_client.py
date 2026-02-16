
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

"""
Client for interacting with Large Language Models (LLMs).
Supports Google Vertex AI, Google AI Studio, and OpenAI.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import List
from cv_maker.models import CVData, JobDescription, Experience, EarlierExperience
from cv_maker.ssl_helpers import configure_ssl_env

# Logger is configured in main.py
logger = logging.getLogger(__name__)

class LLMClient:
    """
    Abstraction layer for LLM providers.
    Handles prompted requests for CV analysis and tailoring.
    """
    def __init__(self, provider: str = "gemini"):
        self.provider = provider
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No API key found. LLM features will fallback to Mock Data.")
        
        self.cache_file = Path("user_content/.model_cache.json")

    def _load_cache(self) -> List[str]:
        """Loads valid models from local cache if less than 24h old."""
        try:
            if not self.cache_file.exists():
                return []
            
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
            
            # Check expiry (24 hours = 86400 seconds)
            if time.time() - data.get("timestamp", 0) > 86400:
                logger.info("Model cache expired.")
                return []
            
            return data.get("models", [])
        except Exception as e:
            logger.warning(f"Failed to load model cache: {e}")
            return []

    def _save_cache(self, models: List[str]):
        """Saves discovered models to local cache."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump({
                    "timestamp": time.time(),
                    "models": models
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save model cache: {e}")

    def _call_llm(self, prompt: str) -> str:
        """
        Mockable wrapper for LLM calls.
        Supports Google Generative AI (Gemini) and OpenAI.
        Falls back to Mock Data if calls fail.
        """
        # Ensure custom CA bundle is visible to httpx-based SDKs
        configure_ssl_env()

        # Define Mock Data inner function for reuse
        def get_mock_data():
            logger.warning("[!] Using MOCK DATA for demonstration (API keys missing or validation failed).")
            return """
            {
                "name": "Jane Doe",
                "title": "Senior Engineer",
                "contact_info": "City, Country | +1 555 0101 | jane@example.com\\ngithub.com/janedoe",
                "executive_summary": "A senior engineer with experience in distributed systems.",
                "competencies": [
                    ["Category A:", "Skill 1, Skill 2."],
                    ["Category B:", "Skill 3, Skill 4."]
                ],
                "experience": [
                    {
                        "company": "COMPANY A",
                        "location": "City, Country",
                        "dates": "Jan 2020 – Present",
                        "title": "Senior Engineer",
                        "summary_italic": "Led key initiatives.",
                        "bullets": [
                            ["Project A:", "Delivered X."],
                            ["Project B:", "Optimized Y."]
                        ]
                    }
                ],
                "projects": [
                    ["Project X:", "Description of project X."],
                    ["Project Y:", "Description of project Y."]
                ],
                "education": ["BS in Computer Science, University Name"],
                "certifications": "Certification A | Certification B"
            }
            """

        if not self.api_key:
            return get_mock_data()
        
        try:
            # 1. Vertex AI (Priority if provider is 'vertex' or configured)
            if self.provider == "vertex" or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                try:
                    import vertexai
                    from vertexai.generative_models import GenerativeModel
                    # Ensure project/location are set in env or via gcloud init
                    vertexai.init() 
                    model = GenerativeModel("gemini-1.5-flash")
                    response = model.generate_content(prompt)
                    return response.text
                except ImportError:
                    logger.warning("google-cloud-aiplatform not installed. Skipping Vertex.")
                except Exception as e:
                    logger.warning(f"Vertex AI failed: {e}. Falling back...")

            # 2. OpenAI
            if self.api_key and self.api_key.startswith("sk-"):
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo", # Fallback model
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                return response.choices[0].message.content

            # 3. Google GenAI (New SDK)
            from google import genai
            
            client = genai.Client(api_key=self.api_key)
            
            # 1. Try Cached Models first
            cached_models = self._load_cache()
            if cached_models:
                logger.info(f"Loaded {len(cached_models)} models from cache.")
                for model_name in cached_models:
                    try:
                        logger.info(f"Attempting cached model: {model_name}")
                        response = client.models.generate_content(
                            model=model_name,
                            contents=prompt
                        )
                        return response.text
                    except Exception as e:
                        logger.warning(f"Cached model {model_name} failed: {e}")
            
            # 2. If Cache failed/empty, try Auto-Discovery
            logger.info("Cache failed or empty. Attempting auto-discovery...")
            try:
                 discovered = self.discover_models(client)
                 
                 # Save discovered to cache immediately so next run is fast
                 if discovered:
                     self._save_cache(discovered)
                 
                 # 3. Try Discovered Models
                 for model_name in discovered:
                     try:
                         logger.info(f"Attempting discovered model: {model_name}")
                         response = client.models.generate_content(
                            model=model_name,
                            contents=prompt
                         )
                         return response.text
                     except Exception as e:
                         # Keep trying others
                         logger.warning(f"Model {model_name} failed: {e}")
            except Exception as e:
                logger.warning(f"Auto-discovery failed: {e}")

            # 4. Last Resort: Hardcoded Fallback (if discovery completely broke)
            fallback_models = ['gemini-1.5-flash', 'gemini-1.5-flash-001', 'gemini-pro']
            logger.info("Discovery failed. Trying hardcoded fallbacks...")
            
            last_exception = None
            for model_name in fallback_models:
                try:
                    logger.info(f"Attempting fallback model: {model_name}")
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    return response.text
                except Exception as e:
                     logger.warning(f"Fallback {model_name} failed: {e}")
                     last_exception = e
            
            if last_exception:
                raise last_exception

        except ImportError as e:
            logger.error(f"Missing dependency for specific provider: {e}")
            return get_mock_data()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return get_mock_data()

    def discover_models(self, client=None) -> List[str]:
        """
        Dynamically finds available Gemini models (Flash/Pro) supporting generateContent.
        """
        if not client:
             if self.provider == "vertex": 
                 return None # Not implemented for vertex dynamic yet
             try:
                 from google import genai
                 if not self.api_key: return None
                 client = genai.Client(api_key=self.api_key)
             except:
                 return None

        try:
            # List models
            # SDK v1beta pattern might differ, but assuming standard list_models
            # The new SDK might use client.models.list()
            # We filter for 'gemini' and 'flash' to be safe, or just return the first valid one
            
            # Note: The google-genai SDK implementation for listing might vary. 
            # We will try a generic approach or catch errors.
            
            # Correct method for new SDK is often client.models.list()
            pager = client.models.list() 
            
            candidates = []
            for model in pager:
                methods = getattr(model, 'supported_actions', [])
                if not methods:
                     methods = getattr(model, 'supported_generation_methods', []) # Fallback check 
                
                # Check for generateContent (case insensitive just in case)
                # In new SDK it might be 'generateContent' strings in the list
                if methods and any("generatecontent" == m.lower() for m in methods):
                     name = model.name.split("/")[-1] 
                     name_lower = name.lower()
                     if "gemini" in name_lower and ("flash" in name_lower or "pro" in name_lower):
                         candidates.append(name)
            
            return candidates

        except Exception as e:
            logger.warning(f"Failed to list models: {e}")
            return []

    def analyze_job_description(self, text: str) -> JobDescription:
        """
        Extracts key skills and summary from raw JD text.
        """
        prompt = f"""
        You are an expert technical recruiter. Analyze the following Job Description.
        Extract a specific 'role_title' (e.g. 'Senior Python Engineer').
        Extract a list of 5-10 'key_skills' (technologies, methodologies) required.
        Provide a 1-sentence 'summary' of the role.
        
        Return ONLY valid JSON in this format:
        {{
            "role_title": "Title",
            "key_skills": ["Skill 1", "Skill 2"],
            "summary": "This role involves..."
        }}

        JOB DESCRIPTION:
        {text[:4000]}  # Truncate to avoid context limits
        """
        json_str = self._clean_json(self._call_llm(prompt))
        try:
            data = json.loads(json_str)
            return JobDescription(
                raw_text=text,
                role_title=data.get("role_title", "Top Candidate"),
                key_skills=data.get("key_skills", []),
                summary=data.get("summary", "")
            )
        except json.JSONDecodeError:
            logger.error("Failed to decode LLM response for JD analysis")
            return JobDescription(raw_text=text)

    def tailor_cv(self, master_cv_text: str, jd: JobDescription, github_context: str = "", summarize_years: int = 10) -> CVData:
        """
        Selects relevant experience from the master CV to match the JD.
        """
        
        github_section = ""
        if github_context:
            github_section = f"\nTECHNICAL PORTFOLIO (GITHUB):\n{github_context}\n"

        # Calculate cutoff year
        import datetime
        current_year = datetime.datetime.now().year
        
        if summarize_years > 0:
            cutoff_year = current_year - summarize_years
            
            # Rule 1: Strict inclusion of recent roles
            rule_1 = f"1. Identify ALL roles ending in {cutoff_year} or later (including 'Present'). You MUST include these in the 'experience' array in FULL DETAIL."
            
            # Rule 6: Strict summarization of older roles
            rule_6 = f"""
            6. CRITICAL: Any role that ended BEFORE {cutoff_year} MUST be placed in the separate 'earlier_experience' array.
               For these older roles: Provide Title, Company. Provide a single detailed summary paragraph. DO NOT include dates.
            """
        else:
            # Default/Disable mode: Focus on relevance
            rule_1 = "1. Select the top most relevant roles from the Master CV. Provide these in FULL DETAIL in the 'experience' array."
            rule_6 = "6. Do NOT use the 'earlier_experience' array. Include all relevant roles in the main 'experience' section."

        prompt = f"""
        You are an expert CV writer. I will provide a MASTER CV text and a target JOB DESCRIPTION.
        Your goal is to tailor the CV content to perfectly match the job.
        
        RULES:
        {rule_1}
        2. For each of these detailed roles, include 5-7 bullet points. Prioritize technical depth, specific metrics, and leadership achievements.
        3. Rewrite the Executive Summary to be targeted, substantial, and authoritative (keep it 4-6 sentences).
        4. Reorder/Select 'Core Competencies' to match the JD, preserving technical specificity.
        5. TONE: Expert, Senior Executive, Technical Leader. Avoid generic fluff. Focus on "what" and "how".
        {rule_6}
        7. Use the TECHNICAL PORTFOLIO to substantiate skills in the 'Projects' or 'Competencies' sections, citing specific repositories where relevant.

        Target Role: {jd.summary}
        Target Skills: {', '.join(jd.key_skills)}
        
        MASTER CV CONTENT:
        {master_cv_text[:100000]}
        
        {github_section}

        Return JSON matching this structure:
        {{
            "name": "Applicant Name",
            "title": "Target Title",
            "contact_info": "Phone | Email",
            "executive_summary": "Tailored summary...",
            "competencies": [["Category", "Skill string"], ...],
            "experience": [
                {{
                    "title": "Role Title",
                    "company": "Company",
                    "location": "Loc",
                    "dates": "Date range",
                    "summary_italic": "Brief role context",
                    "bullets": [["Skill/Outcome", "Description"], ...]
                }}
            ],
            "earlier_experience": [
                {{
                    "title": "Role Title",
                    "company": "Company",
                    "summary": "Detailed summary paragraph..."
                }}
            ],
            "projects": [["Title", "Desc"], ...],
            "education": ["Degree 1"],
            "certifications": "Cert string"
        }}
        """
        json_str = self._clean_json(self._call_llm(prompt))
        
        # Default empty data
        default_data = CVData(name="Candidate", title="N/A", contact_info="", executive_summary="", competencies=[], experience=[], earlier_experience=[], projects=[], education=[], certifications="")

        try:
            raw = json.loads(json_str)
            
            # Helper to ensure 2-element tuple
            def to_tuple_2(item):
                if isinstance(item, list):
                    if len(item) >= 2: return (str(item[0]), str(item[1]))
                    elif len(item) == 1: return (str(item[0]), "")
                    else: return ("", "")
                return (str(item), "")

            # Map raw JSON to dataclasses
            exp_list = []
            for job in raw.get("experience", []):
                # Ensure bullets are list of tuples
                clean_bullets = [to_tuple_2(b) for b in job.get("bullets", [])]
                
                exp_list.append(Experience(
                    title=job.get("title", ""),
                    company=job.get("company", ""),
                    location=job.get("location", ""),
                    dates=job.get("dates", ""),
                    summary_italic=job.get("summary_italic"),
                    bullets=clean_bullets
                ))
            
            # Map Earlier Experience
            earlier_list = []
            for job in raw.get("earlier_experience", []):
                earlier_list.append(EarlierExperience(
                    title=job.get("title", ""),
                    company=job.get("company", ""),
                    summary=job.get("summary", "")
                ))
            
            # Map Competencies to tuples
            comps = [to_tuple_2(c) for c in raw.get("competencies", [])]
            # Map Projects to tuples
            projs = [to_tuple_2(p) for p in raw.get("projects", [])]

            return CVData(
                name=raw.get("name", ""),
                title=raw.get("title", ""),
                contact_info=raw.get("contact_info", ""),
                executive_summary=raw.get("executive_summary", ""),
                competencies=comps,
                earlier_experience=earlier_list,
                experience=exp_list, # Fixed order matching dataclass? No, keyword args are safe.
                projects=projs,
                education=raw.get("education", []),
                certifications=raw.get("certifications", "")
            )

        except Exception as e:
            logger.error(f"Failed to map LLM response to CVData: {e}")
            logger.info(f"Raw response: {json_str}")
            return default_data

    def generate_cover_letter(self, master_cv_text: str, jd: JobDescription) -> str:
        """
        Generates a tailored cover letter based on the JD and Master CV.
        """
        prompt = f"""
        You are an expert executive CV writer. Write a high-impact, professional cover letter for the following role.
        
        TARGET ROLE SUMMARY: {jd.summary}
        KEY SKILLS REQUIRED: {', '.join(jd.key_skills)}
        
        MY BACKGROUND (MASTER CV):
        {master_cv_text[:50000]}
        
        GUIDELINES:
        1. Salutation: "Dear Hiring Manager," (unless a specific name is clear in the JD description).
        2. Hook: Strong opening statement explaining why I am the perfect strategic fit for *this specific* role.
        3. Body: Connect 2-3 specific achievements from my Master CV directly to the Key Skills required.
        4. Tone: Confident, Senior Executive, Concise.
        5. Length: 300-400 words maximum.
        6. Format: Return ONLY the body of the letter. Do not include address blocks (the system handles that). Start with the Salutation.
        """
        return self._call_llm(prompt)

    def _clean_json(self, text: str) -> str:
        """Helper to strip code fences from LLM output"""
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return text.strip()
