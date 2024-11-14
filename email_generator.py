import requests
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
from gnews import GNews
from difflib import SequenceMatcher
from groq import Groq

# Configuration
class APIConfig:
    """Configuration for API keys and endpoints"""
    APOLLO_API_KEY: str = "ZPzDIm0lZSfnXzhMA4fKLw"
    GROQ_API_KEY: str = "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"
    GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"

    @classmethod
    def get_apollo_headers(cls) -> Dict[str, str]:
        return {
            "accept": "application/json",
            "Cache-Control": "no-cache",
            "Content-Type": "application/json",
            "x-api-key": cls.APOLLO_API_KEY
        }

# Data Models
@dataclass
class Article:
    """Data model for news articles"""
    title: str
    description: str
    url: str
    publisher: str
    published_date: str

    @classmethod
    def from_gnews_data(cls, article_data: Dict) -> 'Article':
        return cls(
            title=article_data.get("title", ""),
            description=article_data.get("description", ""),
            url=article_data.get("url", ""),
            publisher=article_data.get("publisher", {}).get("title", ""),
            published_date=article_data.get("published_date", "")
        )

@dataclass
class ProspectProfile:
    """Data model for prospect information"""
    name: str
    email: str
    title: str
    company_name: str
    industry: str
    keywords: List[str]
    technologies: List[str]
    location: Dict[str, str]
    linkedin_url: Optional[str] = None

# Prompt Generation Functions
def _determine_seniority(title: str) -> str:
    """Determine seniority level from title"""
    title_lower = title.lower()
    if any(x in title_lower for x in ['ceo', 'cto', 'cfo', 'founder', 'president', 'director']):
        return 'C-Level/Executive'
    elif any(x in title_lower for x in ['head', 'lead', 'manager', 'principal']):
        return 'Management'
    elif any(x in title_lower for x in ['senior', 'sr', 'architect']):
        return 'Senior'
    return 'Individual Contributor'

def create_article_selection_prompt(prospect: ProspectProfile, articles: List[Article]) -> str:
    """Create prompt for strategic article selection"""
    def _get_business_context(prospect: ProspectProfile) -> str:
        return f"""
        - Company Scale: {prospect.company_name} in {prospect.industry}
        - Technology Focus: Using {', '.join(prospect.technologies[:3])} {'and others' if len(prospect.technologies) > 3 else ''}
        - Geographic Context: Based in {prospect.location.get('city', '')}, {prospect.location.get('country', '')}
        """

    def _format_articles_for_prompt(articles: List[Article]) -> str:
        return "\n".join([
            f"""
            Article {i+1}:
            - Title: {article.title}
            - Description: {article.description}
            - Publisher: {article.publisher}
            - Relevance Indicators:
              * Industry alignment
              * Technology relevance
              * Business impact potential
            """
            for i, article in enumerate(articles)
        ])

    return f"""
    You are a B2B Marketing Strategist specializing in personalized outreach campaigns. Your goal is to select the most strategic article that will:
    1. Build credibility and trust through industry knowledge
    2. Demonstrate understanding of prospect's business challenges
    3. Create a natural conversation opener
    4. Position our company as a thought leader
    5. Provide genuine value before any pitch

    Target Prospect Analysis:
    - Name: {prospect.name}
    - Title: {prospect.title} (Seniority Level: {_determine_seniority(prospect.title)})
    - Company: {prospect.company_name}
    - Industry: {prospect.industry}
    - Technologies Used: {', '.join(prospect.technologies)}
    - Business Context: {_get_business_context(prospect)}

    Selection Criteria Priority:
    1. RELEVANCE: Article should directly relate to:
       - Prospect's industry challenges
       - Their technology stack
       - Current market trends affecting their business
       - Their likely business objectives based on role

    2. TIMELINESS:
       - Recent developments in their industry
       - Emerging trends they should know about
       - Competitive advantages they might be missing

    3. STRATEGIC VALUE:
       - Must provide actionable insights
       - Should align with their decision-making level
       - Could highlight pain points we can address
       - Should position us as knowledgeable in their space

    Available Articles:
    {_format_articles_for_prompt(articles)}

    TASK:
    Select the article that best serves as a strategic conversation opener, considering:
    - The prospect's decision-making authority
    - Current industry challenges
    - Potential business pain points
    - Our ability to add value to the discussion

    Return only the exact title of the most strategic article for engaging this prospect.
    """

def create_email_generation_prompt(
    prospect: ProspectProfile,
    article: Article,
    sender_info: Dict
) -> str:
    """Creates a strategic B2B warmup email prompt"""
    return f"""
    You are an expert B2B Marketing Strategist crafting a high-impact first touch email.

    CAMPAIGN OBJECTIVES:
    1. Primary: Generate qualified leads by establishing meaningful business relationships
    2. Secondary: Build brand awareness and position as industry thought leader
    3. Tertiary: Create engagement through valuable insights sharing

    TARGET OUTCOME:
    - Immediate: Receive a response showing interest in discussing the shared insights
    - Short-term: Schedule a discovery call to explore mutual value
    - Long-term: Develop a business relationship based on trust and value exchange

    PROSPECT PROFILE:
    Decision Maker Details:
    - Name: {prospect.name}
    - Title: {prospect.title}
    - Seniority: {_determine_seniority(prospect.title)}
    - Company: {prospect.company_name}
    - Industry: {prospect.industry}
    - Tech Stack: {', '.join(prospect.technologies)}
    - Location: {prospect.location.get('city', '')}, {prospect.location.get('country', '')}

    CONTENT FOUNDATION:
    Selected Article:
    - Title: {article.title}
    - Description: {article.description}
    - URL: {article.url}
    - Publisher: {article.publisher}

    Sender Authority:
    - Name: {sender_info['name']}
    - Title: {sender_info['title']}
    - Company: {sender_info['company']}
    - LinkedIn: {sender_info.get('linkedin_url', '')}

    EMAIL CRAFTING GUIDELINES:

    1. Psychology & Approach:
       - Build credibility through industry insight
       - Show understanding of their business context
       - Create value before asking for anything
       - Respect their time with conciseness
       - Demonstrate relevance to their role

    2. Structure Requirements:
       a) Opening (2-3 lines):
          - Personalized greeting
          - Immediate value proposition
          - Article reference as conversation starter

       b) Body (3-4 lines):
          - Connect article insights to their business
          - Demonstrate industry understanding
          - Share relevant experience briefly
          - Add unique perspective

       c) Closing (1-2 lines):
          - Soft, engaging call-to-action
          - Focus on value exchange
          - Make response easy

    3. Tone & Style:
       - Professional yet conversational
       - Confident but not aggressive
       - Consultative rather than sales-focused
       - Peer-to-peer communication level

    4. Technical Requirements:
       - Keep under 150 words
       - Use markdown for article link: [Title](URL)
       - Include proper spacing
       - Professional signature block

    5. Avoid:
       - Direct selling
       - Generic statements
       - Long paragraphs
       - Multiple calls-to-action
       - Obvious templated language

    GENERATE: Create a strategic first-touch email that positions us as a valuable thought partner while encouraging engagement around the shared insights.
    Provide only the email content without any introductory lines or explanations. The email should be ready to send as is.
    """

# Core Services
class NewsService:
    """Service for fetching and managing news articles"""
    def __init__(self, language='en', country='US', period='7d', max_results=5):
        self.gnews = GNews(language=language, country=country, period=period, max_results=max_results)
        self.article_cache = {}

    def get_relevant_articles(self, keywords: List[str], company_url: str = None) -> List[Article]:
        try:
            articles = []
            for keyword in keywords:
                articles.extend(self._get_keyword_articles(keyword))
            if company_url:
                articles.extend(self._get_company_articles(company_url))
            return [Article.from_gnews_data(article) for article in articles]
        except Exception as e:
            logger.error(f"Error fetching news articles: {str(e)}")
            raise NewsAPIError(f"Failed to fetch news articles: {str(e)}")

    def _get_keyword_articles(self, keyword: str) -> List[Dict]:
        try:
            if not keyword:
                return []
            cached_articles = self.article_cache.get(keyword)
            if cached_articles:
                return cached_articles
            search_term = f"{keyword} programming" if keyword.lower() == 'python' else keyword
            articles = self.gnews.get_news(search_term)
            if articles:
                self.article_cache[keyword] = articles
            return articles or []
        except Exception as e:
            logger.warning(f"Failed to fetch articles for keyword '{keyword}': {e}")
            return []

    def _get_company_articles(self, company_url: str) -> List[Dict]:
        try:
            if not company_url:
                return []
            cached_articles = self.article_cache.get(company_url)
            if cached_articles:
                return cached_articles
            articles = self.gnews.get_news_by_site(company_url)
            if articles:
                self.article_cache[company_url] = articles
            return articles or []
        except Exception as e:
            logger.warning(f"Failed to fetch company articles: {e}")
            return []

class ApolloService:
    """Service for interacting with Apollo API"""
    def __init__(self):
        self.headers = APIConfig.get_apollo_headers()

    def get_contact_details(self, email: str) -> Optional[Dict]:
        try:
            url = f"https://api.apollo.io/api/v1/people/match?email={email}"
            response = requests.post(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return self._format_contact_details(data.get("person", {}))
        except Exception as e:
            logger.error(f"Apollo API error fetching contact details: {str(e)}")
            raise ApolloAPIError(f"Failed to fetch contact details: {str(e)}")

    def get_company_details(self, domain: str) -> Optional[Dict]:
        try:
            url = f"https://api.apollo.io/api/v1/organizations/enrich?domain={domain}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return self._format_company_details(data.get("organization", {}))
        except Exception as e:
            logger.error(f"Apollo API error fetching company details: {str(e)}")
            raise ApolloAPIError(f"Failed to fetch company details: {str(e)}")

    def _format_contact_details(self, person: Dict) -> Dict:
        if not person:
            return {}
        return {
            "name": person.get("name", ""),
            "email": person.get("email", ""),
            "title": person.get("title", ""),
            "company": person.get("organization", {}).get("name", ""),
            "company_url": person.get("organization", {}).get("website_url", ""),
            "linkedin_url": person.get("linkedin_url", ""),
            "city": person.get("city", ""),
            "state": person.get("state", ""),
            "country": person.get("country", ""),
            "keywords": person.get("organization", {}).get("keywords", [])
        }

    def _format_company_details(self, organization: Dict) -> Dict:
        if not organization:
            return {}
        return {
            "name": organization.get("name", ""),
            "industry": organization.get("industry", ""),
            "website_url": organization.get("website_url", ""),
            "keywords": organization.get("keywords", []),
            "technologies": [
                tech.get("name")
                for tech in organization.get("current_technologies", [])
                if tech.get("name")
            ]
        }

class EmailService:
    """Service for generating personalized emails"""
    def __init__(self, sender_info: Dict):
        self.groq_client = Groq(api_key=APIConfig.GROQ_API_KEY)
        self.sender_info = sender_info

    def generate_email(self, prospect: ProspectProfile, article: Article) -> str:
        try:
            prompt = create_email_generation_prompt(prospect, article, self.sender_info)
            completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert B2B email copywriter."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-70b-versatile"
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating email: {str(e)}")
            raise B2BEmailerError(f"Failed to generate email: {str(e)}")

def generate_warmup_email(prospect_email: str) -> Optional[str]:
    """Main function to generate a personalized warmup email"""
    try:
        # Initialize services
        apollo_service = ApolloService()
        news_service = NewsService()

        # Sender information
        sender_info = {
            "name": "Neeraj Kumar",
            "title": "Founder, CEO",
            "company": "Valuebound",
            "email": "neeraj@valuebound.com",
            "linkedin_url": "http://www.linkedin.com/in/neerajskydiver"
        }

        email_service = EmailService(sender_info)

        # Get prospect information
        contact_details = apollo_service.get_contact_details(prospect_email)
        if not contact_details:
            raise ValueError("Contact details not found")

        company_details = apollo_service.get_company_details(contact_details.get("company_url", ""))
        if not company_details:
            raise ValueError("Company details not found")

        # Create prospect profile
        prospect = ProspectProfile(
            name=contact_details["name"],
            email=prospect_email,
            title=contact_details["title"],
            company_name=company_details["name"],
            industry=company_details["industry"],
            keywords=company_details["keywords"],
            technologies=company_details["technologies"],
            location={
                "city": contact_details["city"],
                "state": contact_details["state"],
                "country": contact_details["country"]
            },
            linkedin_url=contact_details["linkedin_url"]
        )

        # Get relevant articles
        articles = news_service.get_relevant_articles(prospect.keywords, prospect.company_name)
        if not articles:
            raise ValueError("No relevant articles found")

        # Generate and return email
        return email_service.generate_email(prospect, articles[0])

    except Exception as e:
        logger.error(f"Error generating warmup email: {e}")
        return None
