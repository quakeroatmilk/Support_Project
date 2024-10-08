from django.shortcuts import render, redirect
from django.http import JsonResponse
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sqlite3
import pygwalker as pyg
import plotly.express as px
import json
from transformers import pipeline
from collections import Counter
import re
import os
from django.conf import settings
import PyPDF2
import plotly
import pdfplumber
from django.templatetags.static import static
import spacy
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from .models import CorporateInfo, ShareCapital, Directors, Shareholders, FinancialInfo, UserManual
from django.core.files.storage import FileSystemStorage

# Load sentiment analysis model
sentiment_analysis = pipeline('sentiment-analysis')

# Load static business rules data
support_df = pd.DataFrame([
    {"agency": "Suruhanjaya Koperasi Malaysia", "supports": "Grant programs, cooperative development support, advisory services", "business_type": "Cooperatives, SMEs"},
    {"agency": "Institut Koperasi Malaysia", "supports": "Training programs, research and development, capacity building", "business_type": "Cooperatives, SMEs, Startups"},
    {"agency": "SME Corp Malaysia", "supports": "Grants, business development services, advisory services, market access programs", "business_type": "SMEs, Startups, Innovative and High-Growth Businesses"},
    {"agency": "Tekun Nasional", "supports": "Microfinancing, business loans, advisory services, entrepreneurial training", "business_type": "Microenterprises, SMEs, Startups"},
    {"agency": "Institut Keusahawanan Negara", "supports": "Entrepreneurial training, business coaching, incubation programs", "business_type": "Startups, SMEs, Innovative and High-Growth Businesses"},
    {"agency": "Bank Rakyat", "supports": "Business loans, microfinancing, savings and investment services", "business_type": "SMEs, Cooperatives, Individuals"},
    {"agency": "SME Bank", "supports": "Business financing, advisory services, business development programs", "business_type": "SMEs, Startups, Social Enterprises"},
    {"agency": "UDA Holding Berhad", "supports": "Property development support, business premises, advisory services", "business_type": "Property Development Companies, SMEs, Startups"},
    {"agency": "Perbadanan Nasional Berhad", "supports": "Franchise development support, financing for franchise businesses, advisory services", "business_type": "Franchise Businesses, SMEs, Startups"},
    {"agency": "Amanah Ikhtiar Malaysia", "supports": "Microfinancing, entrepreneurial training, capacity building programs", "business_type": "Microenterprises, Women Entrepreneurs, SMEs"},
])

# Load synthetic data for prediction analysis
synthetic_data_path = os.path.join(settings.BASE_DIR, 'static', 'data', 'synthetic_success_rate.csv')
profile_data_path = os.path.join(settings.BASE_DIR,  'static', 'data', 'profile_data.csv')

df = pd.read_csv(synthetic_data_path)
profile_data = pd.read_csv(profile_data_path)

# Data Preprocessing for Prediction Analysis
label_encoders = {}
categorical_columns = ['Business Type', 'Industry', 'Location', 'Number of Trainings', 'Advisory Sessions', 'Microfinancing', 'Business Development Services']

for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

X = df.drop(columns=['Success Rate (%)'])
y = df['Success Rate (%)']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the prediction model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Recommendation System
def recommendation_system(request):
    business_types = support_df['business_type'].str.split(', ').explode().unique()
    business_types_list = business_types.tolist()
    return render(request, 'index.html', {'business_types': business_types_list})

def recommend_supports(business_type):
    relevant_supports = support_df[support_df['business_type'].str.contains(business_type, case=False)]
    
    if relevant_supports.empty:
        return pd.DataFrame(columns=['agency', 'supports', 'business_type', 'logo', 'portal_link'])

    # Add logo and portal link for each agency manually
    relevant_supports['logo'] = relevant_supports['agency'].map({
        'Suruhanjaya Koperasi Malaysia': static('logos/SKM.jpg'),
        'Institut Koperasi Malaysia': static('logos/IKMA.jpg'),
        'SME Corp Malaysia': static('logos/SMECORP.jpg'),
        'Tekun Nasional': static('logos/TEKUN.jpg'),
        'Institut Keusahawanan Negara': static('logos/INSKEN.png'),
        'Bank Rakyat': static('logos/BRAKYAT.jpg'),
        'SME Bank': static('logos/SMEBANK.jpg'),
        'UDA Holding Berhad': static('logos/UDA.png'),
        'Perbadanan Nasional Berhad': static('logos/PERNAS.png'),
        'Amanah Ikhtiar Malaysia': static('logos/AIM.png')
    })


    relevant_supports['portal_link'] = relevant_supports['agency'].map({
        'Suruhanjaya Koperasi Malaysia': 'https://www.skm.gov.my/',
        'Institut Koperasi Malaysia': 'https://www.ikma.edu.my/',
        'SME Corp Malaysia': 'https://www.smecorp.gov.my/',
        'Tekun Nasional': 'https://www.tekun.gov.my/',
        'Institut Keusahawanan Negara': 'https://www.insken.gov.my/',
        'Bank Rakyat': 'https://www.bankrakyat.com.my/',
        'SME Bank': 'https://www.smebank.com.my/',
        'UDA Holding Berhad': 'https://www.uda.com.my/',
        'Perbadanan Nasional Berhad': 'https://pernas.my/',
        'Amanah Ikhtiar Malaysia': 'https://www.aim.gov.my/'
    })

    return relevant_supports

def recommend(request):
    business_type = request.GET.get('business_type')
    recommended_supports = recommend_supports(business_type)
    recommendations = recommended_supports.to_dict(orient='records')
    return JsonResponse(recommendations, safe=False)

# Prediction Analysis
def prediction_analysis(request):
    business_types = label_encoders['Business Type'].inverse_transform(df['Business Type'].unique())
    industries = label_encoders['Industry'].inverse_transform(df['Industry'].unique())
    locations = label_encoders['Location'].inverse_transform(df['Location'].unique())
    
    if request.method == 'POST':
        try:
            input_data = {
                'Business Type': request.POST['business_type'],
                'Industry': request.POST['industry'],
                'Years in Operation': int(request.POST['years_in_operation']),
                'Location': request.POST['location'],
                'Grant Amount': float(request.POST['grant_amount']),
                'Number of Trainings': request.POST['number_of_trainings'],
                'Advisory Sessions': request.POST['advisory_sessions'],
                'Microfinancing': request.POST['microfinancing'],
                'Business Development Services': request.POST['business_dev_services']
            }

            input_df = pd.DataFrame([input_data])

            # Apply label encoding
            for col in categorical_columns:
                if col in input_df.columns:
                    input_df[col] = label_encoders[col].transform(input_df[col])

            # Normalize the features
            input_df = scaler.transform(input_df)

            # Predict success rate
            prediction = model.predict(input_df)[0]

            return redirect(f"/prediction-result/?prediction={prediction}")
        
        except Exception as e:
            return render(request, 'prediction_analysis.html', {'error': str(e)})

    return render(request, 'prediction_analysis.html', {
        'business_types': business_types,
        'industries': industries,
        'locations': locations
    })

# Prediction Result View
def prediction_result(request):
    prediction = request.GET.get('prediction', None)

    if prediction is not None:
        prediction = float(prediction)
    else:
        prediction = None

    return render(request, 'prediction_result.html', {
        'prediction': prediction
    })

# Profile Data Analytics
def profile_analytics(request):
    # Path to your CSV file
    profile_data_path = os.path.join(settings.BASE_DIR, 'static', 'data', 'profile_data.csv')
    
    # Read the CSV into a DataFrame
    profile_data = pd.read_csv(profile_data_path)
    
    # Generate PygWalker with default specifications
    walker = pyg.walk(
        profile_data,
        spec={
            "encoding": {
                "x": {
                    "field": "Location",  # X-axis as Location
                    "type": "nominal",    # Treat Location as categorical/nominal data
                },
                "y": {
                    "field": "Amount",     # Y-axis as Amount
                    "type": "quantitative",  # Treat Amount as numeric/quantitative data
                    "aggregate": "sum",     # Sum of Amount
                },
                "color": {
                    "field": "Location",  # Color bars based on Location
                    "type": "nominal",    # Nominal data for colors
                }
            },
            "mark": "bar"  # Set default chart type to bar chart
        }
    )

    # Convert the PygWalker chart to HTML so it can be embedded in the template
    walker_html = walker.to_html()

    # Render the template with the PygWalker chart
    return render(request, 'profile_analytics.html', {'pygwalker': walker_html})


# User Feedback Submission
def display_feedback(request):
    feedback_db_path = os.path.join(settings.BASE_DIR, 'static', 'data', 'feedback.db')
    msg = ""
    if request.method == 'POST':
        try:
            name = request.POST['name']
            email = request.POST['email']
            feedback_type = request.POST['feedback_type']
            rating = request.POST['rating']
            comments = request.POST['comments']

            sentiment_result = sentiment_analysis(comments)[0]
            sentiment = sentiment_result['label']

            with sqlite3.connect(feedback_db_path) as con:
                cur = con.cursor()
                cur.execute("INSERT INTO feedback (name, email, feedback_type, rating, comments, sentiment) VALUES (?, ?, ?, ?, ?, ?)", 
                            (name, email, feedback_type, rating, comments, sentiment))
                con.commit()
                msg = "Feedback submitted successfully."
            return redirect('view_feedback')

        except Exception as e:
            msg = f"Error occurred: {e}"

    return render(request, 'user_feedback.html', {'msg': msg})

# View Feedback (Sentiment Analysis)
def view_feedback(request):
    feedback_db_path = os.path.join(settings.BASE_DIR, 'static', 'data', 'feedback.db')
    
    con = sqlite3.connect(feedback_db_path)
    feedback_df = pd.read_sql_query("SELECT * from feedback", con)
    con.close()

    feedback_df['sentiment'] = feedback_df['comments'].apply(lambda x: sentiment_analysis(x)[0]['label'])

    sentiment_counts = feedback_df['sentiment'].value_counts(normalize=True) * 100
    average_rating = feedback_df['rating'].mean()
    rounded_average_rating = round(average_rating, 2)

    top_positive_comments = feedback_df[feedback_df['sentiment'] == 'POSITIVE'].nlargest(3, 'rating')['comments'].tolist()
    top_negative_comments = feedback_df[feedback_df['sentiment'] == 'NEGATIVE'].nsmallest(3, 'rating')['comments'].tolist()

    sentiment_bar_chart = px.bar(
        x=sentiment_counts.index.tolist(),
        y=sentiment_counts.values.tolist(),
        labels={'x': 'Sentiment', 'y': 'Percentage'},
        title="Sentiment Analysis by Feedback Type",
        color_discrete_sequence=['#2ca02c', '#d62728', '#1f77b4']
    )
    sentiment_graphJSON = json.dumps(sentiment_bar_chart, cls=plotly.utils.PlotlyJSONEncoder)

    rating_counts = feedback_df['rating'].value_counts().sort_index()
    rating_bar_chart = px.bar(
        x=rating_counts.index.tolist(),
        y=rating_counts.values.tolist(),
        labels={'x': 'Rating', 'y': 'Count'},
        title="Rating Summary",
        color_discrete_sequence=['#1f77b4']
    )
    rating_graphJSON = json.dumps(rating_bar_chart, cls=plotly.utils.PlotlyJSONEncoder)

    return render(request, 'view_feedback.html', {
        'sentiment_graphJSON': sentiment_graphJSON,
        'rating_graphJSON': rating_graphJSON,
        'top_positive_comments': top_positive_comments,
        'top_negative_comments': top_negative_comments,
        'average_rating': rounded_average_rating
    })

# Functions for PDF and Company Analysis
from datetime import datetime

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using pdfplumber and Tesseract OCR as fallback."""
    extracted_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
    except Exception:
        pass

    # Use OCR if no text found
    if not extracted_text.strip():
        try:
            images = convert_from_path(pdf_path, dpi=300)
            for image in images:
                text = pytesseract.image_to_string(image, config='--psm 6')
                extracted_text += text + "\n"
        except Exception:
            pass

    return extracted_text

def analyze_extracted_text(text):
    """Analyze and extract corporate and financial information from extracted PDF text."""
    parsed_data = {'corporate_info': {}, 'financial_info': {}}

    # Define regex patterns for corporate info
    corporate_patterns = {
        'company_name': r"Name\s*[:\-\–]?\s*(.+)",
        'company_number': r"Registration \s*No\s*[:\-\–]?\s*(.+)",
        'incorporation_date': r"Incorporation\s*Date\s*[:\-\–]?\s*(\d{2}-\d{2}-\d{4})",
        'registration_date': r"Registration\s*Date\s*[:\-\–]?\s*(\d{2}-\d{2}-\d{4})",
        'registered_address': r"Registered\s*Address\s*[:\-\–]?\s*(.+)",
        'business_address': r"Business\s*Address\s*[:\-\–]?\s*(.+)",
        'nature_of_business': r"Nature\s*of\s*Business\s*[:\-\–]?\s*(.+)"
    }

    # Apply regex to extract corporate info
    for line in text.splitlines():
        for key, pattern in corporate_patterns.items():
            match = re.search(pattern, line, re.IGNORECASE)
            if match and not parsed_data['corporate_info'].get(key):
                parsed_data['corporate_info'][key] = match.group(1).strip()

    # Financial info patterns
    financial_patterns = {
        'non_current_assets': r"Non-current\s*assets\s*[:\-]?\s*([\d,\.]+)",
        'current_assets': r"Current\s*assets\s*[:\-]?\s*([\d,\.]+)",
        'non_current_liabilities': r"Non-current\s*liabilities\s*[:\-]?\s*([\d,\.]+)",
        'current_liabilities': r"Current\s*liabilities\s*[:\-]?\s*([\d,\.]+)",
        'share_capital': r"Share\s*Capital\s*[:\-]?\s*([\d,\.]+)",
        'retained_earnings': r"Retained\s*Earnings\s*[:\-]?\s*([\d,\.]+)",
        'revenue': r"Revenue\s*[:\-]?\s*([\d,\.]+)"
    }

    for key, pattern in financial_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        parsed_data['financial_info'][key] = match.group(1).strip() if match else "Not found"

    return parsed_data

def calculate_company_age_and_evaluation(incorporation_date_str):
    """Calculate the company age and evaluate it based on incorporation date."""
    try:
        incorporation_date = datetime.strptime(incorporation_date_str, "%d-%m-%Y")
        current_date = datetime.now()
        age_in_years = current_date.year - incorporation_date.year
        if (current_date.month, current_date.day) < (incorporation_date.month, incorporation_date.day):
            age_in_years -= 1

        # Evaluate company based on age
        if age_in_years < 5:
            evaluation = "Young Startup"
        elif 5 <= age_in_years <= 10:
            evaluation = "Growing Business"
        else:
            evaluation = "Established Company"

        return age_in_years, evaluation
    except ValueError:
        return "Invalid Date", "Unknown"

def organization_analytic(request):
    """Process uploaded PDF to extract company information and evaluate company age."""
    analysis = None

    if request.method == 'POST':
        if 'pdf_file' in request.FILES:
            pdf_file = request.FILES['pdf_file']
            pdf_dir = os.path.join(settings.BASE_DIR, 'uploads', 'pdfs')
            os.makedirs(pdf_dir, exist_ok=True)

            pdf_path = os.path.join(pdf_dir, pdf_file.name)
            with open(pdf_path, 'wb+') as destination:
                for chunk in pdf_file.chunks():
                    destination.write(chunk)

            # Extract text from the uploaded PDF
            text = extract_text_from_pdf(pdf_path)

            if text:
                analysis = analyze_extracted_text(text)

                # Calculate company age and evaluation
                incorporation_date = analysis['corporate_info'].get('incorporation_date', None)
                company_age, company_evaluation = calculate_company_age_and_evaluation(incorporation_date)

                # Save company info and financial info
                if 'company_name' in analysis['corporate_info']:
                    CorporateInfo.objects.create(**analysis['corporate_info'])

                FinancialInfo.objects.create(**analysis['financial_info'])

                # Add company age and evaluation to the result
                analysis['company_age'] = company_age
                analysis['company_evaluation'] = company_evaluation

                # Redirect to the result page and pass the result data
                return render(request, 'organization_analytics.html', {'analysis': analysis})
            else:
                return render(request, 'upload_pdf.html', {'error': 'PDF extraction failed.'})
        else:
            return render(request, 'upload_pdf.html', {'error': 'No PDF file uploaded.'})
    return render(request, 'upload_pdf.html')

# View for the User Manual
def user_manual(request):
    print("User manual view called")  # Debugging line
    manual_sections = [
        {
            "title": "1. Recommendation System",
            "content": """
                This module provides personalized recommendations based on the selected business type.
                
                <ul>
                    <li>Navigate to the <strong>Recommendation</strong> tab.</li>
                    <li>Choose a business type from the drop-down menu (e.g., SMEs, Startups, Cooperatives).</li>
                    <li>Click on <strong>Get Recommendations</strong>.</li>
                </ul>
                
            """,
            "images": [
                request.build_absolute_uri(static("images/rec_1.png")),
                request.build_absolute_uri(static("images/rec_2.png")),
                request.build_absolute_uri(static("images/rec_3.png")),
            ]
        },
        {
            "title": "2. Prediction Analytics",
            "content": """
                This module predicts your business's success rate based on the support received.
                
                <ul>
                    <li>Navigate to the Prediction Analysis tab.</li>
                    <li>Fill in the required details such as:</li>
                        <ul>
                            <li>Business Type</li>
                            <li>Industry</li>
                            <li>Grant Amount</li>
                            <li>Number of Trainings</li>
                            <li>Advisory Sessions, etc.</li>
                        </ul>
                    <li>Click on Submit to get the predicted success rate.</li>
                    <li>The system will display the predicted success rate, suggestions and how to improve on the Prediction Result page.</li>
                </ul>
                
            """,
            "images": [
                request.build_absolute_uri(static("images/pre_1.png")),
                request.build_absolute_uri(static("images/pre_2.png")),
                request.build_absolute_uri(static("images/pre_3.png")),
                request.build_absolute_uri(static("images/pre_4.png")),
            ]
        },
        {
            "title": "3. Organization Analytics",
            "content": """
                This module extracts and analyzes corporate and financial data from uploaded PDFs.
                
                <ul>
                    <li>Navigate to the Organization Analytics tab.</li>
                    <li>Upload a PDF containing your business’s corporate information or financial statements.</li>
                    <li>The system also calculates the age of the company and provides an evaluation (e.g., Young Startup, Growing Business, Established Company).</li>
                    <li>The system will extract key data such as:</li>
                        <ul>
                            <li>Company Name</li>
                            <li>Incorporation Date</li>
                            <li>Share Capital</li>
                            <li>Revenue, Assets, Liabilities, etc.</li>
                        </ul> 
                    <li>You can review the extracted data on the results page after the PDF upload.</li>                       
                </ul>
                
            """,
            "images": [
                request.build_absolute_uri(static("images/org_1.png")),
                request.build_absolute_uri(static("images/org_2.png")),
                request.build_absolute_uri(static("images/org_3.png")),
                request.build_absolute_uri(static("images/org_4.png")),
            ]
        },
        {
            "title": "4. Data Analytics",
            "content": """
                This module provides visual analytics of the profile data collected by the system.
                
                <ul>
                    <li>Navigate to the Data Analytics tab.</li>
                    <li>The system will display an interactive chart showing the total amount of support received by location and agency.</li>
                    <li>You can filter the data using the options provided by the chart interface to explore different aspects of the data.</li>                 
                </ul>
                
            """,
            "images": [
                request.build_absolute_uri(static("images/data_1.png")),
                request.build_absolute_uri(static("images/data_2.png")),
            ]
        },
        {
            "title": "5. User Feedback & Sentiment Analysis",
            "content": """
                You can provide feedback on the recommendations and predictions received.
                
                <ul>
                    <li>Navigate to the User Feedback tab.</li>
                    <li>Fill in your name, email, feedback type (e.g., Recommendation, Prediction), rating, and comments.</li>
                    <li>Click on Submit Feedback.</li>
                    <li>The system will analyze your feedback sentiment (Positive, Negative, Neutral) using AI.</li> 
                    <li>You can view the sentiment analysis results and top feedbacks on the View Feedback page. Bar charts display the sentiment percentages, and you can also see the average user rating.</li>                 
                </ul>
                
            """,
            "images": [
                request.build_absolute_uri(static("images/feed_1.png")),
                request.build_absolute_uri(static("images/feed_2.png")),
                request.build_absolute_uri(static("images/feed_3.png")),
            ]
        }
    ]
    print(manual_sections)
    return render(request, 'user_manual.html', {'manual_sections': manual_sections})

from django.http import HttpResponse
from django.template.loader import get_template
from xhtml2pdf import pisa

def render_to_pdf(template_src, context_dict={}):
    template = get_template(template_src)
    html = template.render(context_dict)
    result = HttpResponse(content_type='application/pdf')
    pisa_status = pisa.CreatePDF(html, dest=result)
    if pisa_status.err:
        return HttpResponse('We had some errors <pre>' + html + '</pre>')
    return result

def download_user_manual_pdf(request):
    # Define the sections of the user manual (similar to what is displayed on the HTML page)
    manual_sections = [
        {
            "title": "1. Recommendation System",
            "content": """
                This module provides personalized recommendations based on the selected business type.
                
                <ul>
                    <li>Navigate to the Recommendation tab.</li>
                    <li>Choose a business type from the drop-down menu (e.g., SMEs, Startups, Cooperatives).</li>
                    <li>Click on Get Recommendations.</li>
                </ul>
                
            """,
            "images": [
                request.build_absolute_uri(static("images/rec_1.png")),
                request.build_absolute_uri(static("images/rec_2.png")),
                request.build_absolute_uri(static("images/rec_3.png")),
            ]
        },
        {
            "title": "2. Prediction Analytics",
            "content": """
                This module predicts your business's success rate based on the support received.
                
                <ul>
                    <li>Navigate to the Prediction Analysis tab.</li>
                    <li>Fill in the required details such as:</li>
                        <ul>
                            <li>Business Type</li>
                            <li>Industry</li>
                            <li>Grant Amount</li>
                            <li>Number of Trainings</li>
                            <li>Advisory Sessions, etc.</li>
                        </ul>
                    <li>Click on Submit to get the predicted success rate.</li>
                    <li>The system will display the predicted success rate, suggestions and how to improve on the Prediction Result page.</li>
                </ul>
                
            """,
            "images": [
                request.build_absolute_uri(static("images/pre_1.png")),
                request.build_absolute_uri(static("images/pre_2.png")),
                request.build_absolute_uri(static("images/pre_3.png")),
                request.build_absolute_uri(static("images/pre_4.png")),
            ]
        },
        {
            "title": "3. Organization Analytics",
            "content": """
                This module extracts and analyzes corporate and financial data from uploaded PDFs.
                
                <ul>
                    <li>Navigate to the Organization Analytics tab.</li>
                    <li>Upload a PDF containing your business’s corporate information or financial statements.</li>
                    <li>The system also calculates the age of the company and provides an evaluation (e.g., Young Startup, Growing Business, Established Company).</li>
                    <li>The system will extract key data such as:</li>
                        <ul>
                            <li>Company Name</li>
                            <li>Incorporation Date</li>
                            <li>Share Capital</li>
                            <li>Revenue, Assets, Liabilities, etc.</li>
                        </ul> 
                    <li>You can review the extracted data on the results page after the PDF upload.</li>                       
                </ul>
                
            """,
            "images": [
                request.build_absolute_uri(static("images/org_1.png")),
                request.build_absolute_uri(static("images/org_2.png")),
                request.build_absolute_uri(static("images/org_3.png")),
                request.build_absolute_uri(static("images/org_4.png")),
            ]
        },
        {
            "title": "4. Data Analytics",
            "content": """
                This module provides visual analytics of the profile data collected by the system.
                
                <ul>
                    <li>Navigate to the Data Analytics tab.</li>
                    <li>The system will display an interactive chart showing the total amount of support received by location and agency.</li>
                    <li>You can filter the data using the options provided by the chart interface to explore different aspects of the data.</li>                 
                </ul>
                
            """,
            "images": [
                request.build_absolute_uri(static("images/data_1.png")),
                request.build_absolute_uri(static("images/data_2.png")),
            ]
        },
        {
            "title": "5. User Feedback & Sentiment Analysis",
            "content": """
                You can provide feedback on the recommendations and predictions received.
                
                <ul>
                    <li>Navigate to the User Feedback tab.</li>
                    <li>Fill in your name, email, feedback type (e.g., Recommendation, Prediction), rating, and comments.</li>
                    <li>Click on Submit Feedback.</li>
                    <li>The system will analyze your feedback sentiment (Positive, Negative, Neutral) using AI.</li> 
                    <li>You can view the sentiment analysis results and top feedbacks on the View Feedback page. Bar charts display the sentiment percentages, and you can also see the average user rating.</li>                 
                </ul>
                
            """,
            "images": [
                request.build_absolute_uri(static("images/feed_1.png")),
                request.build_absolute_uri(static("images/feed_2.png")),
                request.build_absolute_uri(static("images/feed_3.png")),
            ]
        }
    ]

    # Prepare context for the PDF template
    context = {'manual_sections': manual_sections}
    
    # Render the PDF
    pdf = render_to_pdf('user_manual_pdf.html', context)
    return pdf

def pdf_download_page(request):
    # You can pass any context if needed
    return render(request, 'pdf_download_page.html', {})







