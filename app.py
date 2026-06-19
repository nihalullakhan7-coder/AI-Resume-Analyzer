import streamlit as st
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt

from skills import skills_db

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Resume Analyzer",
    layout="wide"
)

st.title("AI-Powered Resume Screening and Job Matching System")

st.markdown(
    "Analyze resumes, identify skill gaps, and evaluate job compatibility using NLP."
)

# ----------------------
# ROLE DATABASE
# ----------------------

role_skills = {
    "Machine Learning Engineer": [
        "python",
        "machine learning",
        "deep learning",
        "pytorch"
    ],

    "Data Analyst": [
        "python",
        "sql",
        "pandas",
        "numpy"
    ],

    "Full Stack Developer": [
        "html",
        "css",
        "react",
        "node.js",
        "mysql"
    ],

    "Python Developer": [
        "python",
        "sql",
        "git"
    ]
}

# ----------------------
# SIDEBAR
# ----------------------

st.sidebar.header("Job Setup")

selected_role = st.sidebar.selectbox(
    "Select Target Role",
    list(role_skills.keys())
)

job_description = st.sidebar.text_area(
    "Paste Job Description (Optional)"
)

# ----------------------
# FILE UPLOAD
# ----------------------

uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

# ----------------------
# MAIN LOGIC
# ----------------------

if uploaded_file:

    try:

        text = ""

        with pdfplumber.open(uploaded_file) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text

        text = text.lower()

        # ----------------------
        # SKILL DETECTION
        # ----------------------

        found_skills = []

        for skill in skills_db:

            if skill.lower() in text:
                found_skills.append(skill)

        # ----------------------
        # RESUME SCORE
        # ----------------------

        score = min(len(found_skills) * 7, 100)

        required_skills = role_skills[selected_role]

        missing_role_skills = [

            skill
            for skill in required_skills

            if skill not in found_skills
        ]

        # ----------------------
        # DASHBOARD METRICS
        # ----------------------

        st.subheader("Dashboard")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Skills Found",
                len(found_skills)
            )

        with col2:
            st.metric(
                "Missing Skills",
                len(missing_role_skills)
            )

        with col3:
            st.metric(
                "Resume Score",
                score
            )

        # ----------------------
        # SCORE
        # ----------------------

        st.subheader("Resume Score")

        st.progress(score / 100)

        st.success(f"{score}/100")

        # ----------------------
        # DETECTED SKILLS
        # ----------------------

        st.subheader("Detected Skills")

        if found_skills:

            for skill in found_skills:
                st.success(skill.title())

        else:

            st.warning(
                "No matching skills found."
            )

        # ----------------------
        # RECOMMENDED SKILLS
        # ----------------------

        missing_skills = [

            skill
            for skill in skills_db

            if skill not in found_skills
        ]

        st.subheader("Recommended Skills")

        st.write(missing_skills[:5])

        # ----------------------
        # ANALYTICS CHART
        # ----------------------

        st.subheader("Skill Analytics")

        if found_skills:

            skill_df = pd.DataFrame({

                "Skill": found_skills,

                "Count": [1] * len(found_skills)
            })

            fig, ax = plt.subplots()

            ax.bar(
                skill_df["Skill"],
                skill_df["Count"]
            )

            plt.xticks(rotation=45)

            st.pyplot(fig)

        # ----------------------
        # ROLE ANALYSIS
        # ----------------------

        st.subheader("Target Role")

        st.info(selected_role)

        st.subheader("Skill Gap Analysis")

        if missing_role_skills:

            st.error(
                "Missing Skills: "
                + ", ".join(missing_role_skills)
            )

        else:

            st.success(
                "Great! You match all core skills."
            )
        # ----------------------
        # JOB DESCRIPTION MATCHING
        # ----------------------

        if job_description:

            st.subheader("Job Match Score")

            vectorizer = TfidfVectorizer()

            vectors = vectorizer.fit_transform(
                [
                    text,
                    job_description.lower()
                ]
            )

            similarity = cosine_similarity(
                vectors[0],
                vectors[1]
            )[0][0]

            match_score = round(
                similarity * 100,
                2
            )

            st.progress(
                match_score / 100
            )

            st.success(
                f"{match_score}% Match"
            )

        # ----------------------
        # DOWNLOAD REPORT
        # ----------------------

        report = f"""
AI RESUME ANALYSIS REPORT

Resume Score: {score}/100

Target Role:
{selected_role}

Detected Skills:
{', '.join(found_skills)}

Recommended Skills:
{', '.join(missing_skills[:5])}

Missing Role Skills:
{', '.join(missing_role_skills)}
"""

        st.download_button(
            label="📥 Download Analysis Report",
            data=report,
            file_name="resume_analysis_report.txt",
            mime="text/plain"
        )

    except Exception as e:

        st.error(
            f"Error processing resume: {e}"
        )