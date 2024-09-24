import csv
from django.core.exceptions import ValidationError
from mysupportapp.models import Feedback

csv_file_path = 'feedback.csv'

with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        feedback = Feedback(
            name=row['name'],
            email=row['email'],
            feedback_type=row['feedback_type'],
            rating=int(row['rating']),
            comments=row['comments'],
            sentiment=row['sentiment']
        )
        try:
            feedback.full_clean()  # Validates the model instance
            feedback.save()  # Saves the instance if valid
        except ValidationError as e:
            print(f"Error saving feedback from {row['email']}: {e}")
