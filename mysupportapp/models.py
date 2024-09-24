from django.db import models
import sqlite3
print(sqlite3.sqlite_version)

# Model for storing feedback data (feedback.db)
class Feedback(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    feedback_type = models.CharField(max_length=100)
    rating = models.IntegerField()
    comments = models.TextField()
    sentiment = models.CharField(max_length=100,default='Positive') # Add timestamp when feedback is submitted

    def __str__(self):
        return f"{self.name} - {self.feedback_type} ({self.rating}/5)"


# Model for storing organization data (organization_data.db)
class Organization(models.Model):
    name = models.CharField(max_length=255)
    registration_no = models.CharField(max_length=100)
    business_address = models.TextField()
    nature_of_business = models.TextField()
    total_issued = models.CharField(max_length=100)
    registrar_of_companies = models.CharField(max_length=255, blank=True, null=True)  # Optional field

    def __str__(self):
        return self.name
    
class Agency(models.Model):
    name = models.CharField(max_length=200)
    supports = models.TextField()
    business_type = models.CharField(max_length=300)
    portal_link = models.URLField()
    logo = models.ImageField(upload_to='logos/')

    def __str__(self):
        return self.name
    
class CorporateInfo(models.Model):
    company_name = models.CharField(max_length=255)
    company_number = models.CharField(max_length=50)
    incorporation_date = models.CharField(max_length=50)
    registration_date = models.CharField(max_length=50)
    company_type = models.CharField(max_length=100)
    status = models.CharField(max_length=100)
    registered_address = models.TextField()
    business_address = models.TextField()
    nature_of_business = models.TextField()

class ShareCapital(models.Model):
    company_name = models.CharField(max_length=255)
    total_issued = models.CharField(max_length=100)

class Directors(models.Model):
    company_name = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    registration_no = models.CharField(max_length=50)
    ic_passport_no = models.CharField(max_length=50)
    address = models.TextField()
    designation = models.CharField(max_length=100)
    appointment_date = models.CharField(max_length=50)

class Shareholders(models.Model):
    company_name = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    ic_passport_no = models.CharField(max_length=50)
    total_shares = models.IntegerField()

class FinancialInfo(models.Model):
    company_name = models.CharField(max_length=255)
    non_current_assets = models.CharField(max_length=100)
    current_assets = models.CharField(max_length=100)
    non_current_liabilities = models.CharField(max_length=100)
    current_liabilities = models.CharField(max_length=100)
    share_capital = models.CharField(max_length=100)
    retained_earnings = models.CharField(max_length=100)
    revenue = models.CharField(max_length=100)
    profit_before_tax = models.CharField(max_length=100)
    profit_after_tax = models.CharField(max_length=100)
