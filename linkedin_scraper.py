import os
import requests
import argparse
import logging
import json
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
rapidapi_key = os.getenv('RAPIDAPI_KEY')

def main(profile_url):
    formatted_text, profile_image_url = scrape_linkedin_profile(profile_url)
    
    if formatted_text:  
        logging.info("✅ Profile Data Scraped Successfully:")
        logging.info("\n" + formatted_text)
        if profile_image_url:
            logging.info(f"📸 Profile Image URL: {profile_image_url}")
        else:
            logging.info("❌ No Profile Image URL found.")
    else:
        logging.error("⚠️ Failed to scrape profile data. Profile may be private or API is restricted.")

def scrape_linkedin_profile(profile_url):
    username = extract_username(profile_url)
    
    if not rapidapi_key:
        logging.error("❌ RapidAPI key not found. Set the RAPIDAPI_KEY environment variable.")
        return None, None

    url = "https://linkedin-data-api.p.rapidapi.com/get-profile-posts?"
    querystring = {"username": username}

    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": "linkedin-data-api.p.rapidapi.com"
    }

    try:
        response = requests.get(url, headers=headers, params=querystring)

        if response.status_code == 200:
            profile_data = response.json()
            
            # Save API response for debugging
            with open("linkedin_response.json", "w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=4)
            
            logging.debug(f"🔍 API Response Saved: linkedin_response.json")

            formatted_text, profile_image_url = format_data_for_gpt(profile_data)

            return formatted_text, profile_image_url
        else:
            logging.error(f"❌ Failed to fetch profile data. HTTP Status Code: {response.status_code}")
            return None, None
    except Exception as e:
        logging.exception("⚠️ An error occurred while fetching the profile data.")
        return None, None

def extract_username(linkedin_url):
    """Extracts username from LinkedIn profile URL"""
    parts = linkedin_url.strip("/").split("/")
    return parts[-1] if parts else None

def safe_get_list(data, key):
    """Returns a list from the dictionary, or an empty list if the key is missing."""
    return data.get(key, []) if isinstance(data.get(key), list) else []

def safe_get_value(data, key, default=''):
    """Returns a value from the dictionary or a default value if missing."""
    return data.get(key, default) if data.get(key) is not None else default

def format_data_for_gpt(profile_data):
    try:
        # Ensure profile data exists
        if "data" not in profile_data or not profile_data["data"]:
            logging.error("No profile data found in API response.")
            return None, None

        author_data = profile_data["data"][0].get("author", {})

        # Extract Personal Information
        first_name = safe_get_value(author_data, 'firstName', 'No first name')
        last_name = safe_get_value(author_data, 'lastName', 'No last name')
        full_name = f"{first_name} {last_name}"
        headline = safe_get_value(author_data, 'headline', 'No headline provided')
        profile_url = safe_get_value(author_data, 'url', 'No profile URL provided')

        # Extract Profile Image (Highest Resolution)
        profile_pictures = safe_get_list(author_data, 'profilePictures')
        profile_image_url = profile_pictures[-1]["url"] if profile_pictures else "No profile image found"

        # Extract Summary & Location
        summary = safe_get_value(profile_data["data"][0], 'text', 'No summary provided')
        location = safe_get_value(profile_data["data"][0].get('geo', {}), 'full', 'No location provided')

        formatted_text = (
            f"👤 Name: {full_name}\n"
            f"💼 Headline: {headline}\n"
            f"📍 Location: {location}\n"
            f"🔗 Profile: {profile_url}\n"
            f"📝 Summary: {summary}\n"
        )

        # ✅ Extract Experience
        formatted_text += "📌 Experience:\n"
        experience_list = safe_get_list(profile_data["data"][0], 'experience')
        if experience_list:
            for position in experience_list:
                company = safe_get_value(position, 'companyName', 'No company name')
                title = safe_get_value(position, 'title', 'No title')
                job_location = safe_get_value(position, 'location', 'No location')
                job_description = safe_get_value(position, 'description', 'No description').replace('\n', ' ')
                formatted_text += f"- {title} at {company}, {job_location}. {job_description}\n"
        else:
            formatted_text += "- No experience provided\n"

        # ✅ Extract Education
        formatted_text += "🎓 Education:\n"
        education_list = safe_get_list(profile_data["data"][0], 'education')
        if education_list:
            for education in education_list:
                school = safe_get_value(education, 'schoolName', 'No school name')
                degree = safe_get_value(education, 'degree', 'No degree')
                field = safe_get_value(education, 'fieldOfStudy', 'No field of study')
                grade = safe_get_value(education, 'grade', 'No grade')
                edu_description = safe_get_value(education, 'description', 'No description').replace('\n', ' ')
                formatted_text += f"- {degree} in {field} from {school}, Grade: {grade}. {edu_description}\n"
        else:
            formatted_text += "- No education provided\n"

        # ✅ Extract Skills
        formatted_text += "🛠 Skills:\n"
        skills_list = safe_get_list(profile_data["data"][0], 'skills')
        if skills_list:
            for skill in skills_list:
                skill_name = safe_get_value(skill, 'name', 'No skill name')
                formatted_text += f"- {skill_name}\n"
        else:
            formatted_text += "- No skills provided\n"

        # ✅ Extract Languages
        formatted_text += "🗣 Languages:\n"
        languages_list = safe_get_list(profile_data["data"][0], 'languages')
        if languages_list:
            for language in languages_list:
                lang_name = safe_get_value(language, 'name', 'No language name')
                proficiency = safe_get_value(language, 'proficiency', 'No proficiency level')
                formatted_text += f"- {lang_name} ({proficiency})\n"
        else:
            formatted_text += "- No languages provided\n"

        # ✅ Extract Certifications
        formatted_text += "📜 Certifications:\n"
        certifications_list = safe_get_list(profile_data["data"][0], 'certifications')
        if certifications_list:
            for certification in certifications_list:
                cert_name = safe_get_value(certification, 'name', 'No certification name')
                formatted_text += f"- {cert_name}\n"
        else:
            formatted_text += "- No certifications provided\n"

        return formatted_text, profile_image_url

    except Exception as e:
        logging.exception("⚠️ An error occurred during data formatting.")
        return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🔎 Scrape LinkedIn profile data.")
    parser.add_argument("profile_url", help="🔗 LinkedIn profile URL to scrape.")
    args = parser.parse_args()

    main(args.profile_url)
