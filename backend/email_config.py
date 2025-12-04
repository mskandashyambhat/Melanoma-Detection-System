# Email Configuration for Real Doctor Email Sending
# To enable email sending for the 4 real doctors, configure these settings:

# IMPORTANT: For Gmail, you need to create an "App Password"
# Steps:
# 1. Go to Google Account settings: https://myaccount.google.com/
# 2. Security → 2-Step Verification (enable if not already)
# 3. Security → App passwords
# 4. Generate a new app password for "Mail"
# 5. Copy the 16-character password

# Email settings
SENDER_EMAIL = "melanomaresearchlab@gmail.com"  # Your Gmail address
SENDER_PASSWORD = "nrry etkg piuf deon"  # Gmail App Password

# SMTP Configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Real doctor email addresses (already configured in app.py)
# Dr. Shyam (ID 107): skandashyam102@gmail.com
# Dr. Kaushik (ID 108): kaushikmuliya@gmail.com
# Dr. Paavani (ID 109): kpaavani20@gmail.com
# Dr. Deepa (ID 110): deepamk725@gmail.com

# WhatsApp Configuration
# Using pywhatkit (sends via WhatsApp Web - no extra number needed!)
# Sender will appear as "Melanoma Detection Lab" in your WhatsApp profile
WHATSAPP_SENDER_NAME = "Melanoma Detection Lab"

# Real doctor WhatsApp numbers (with country code, format: +919876543210)
REAL_DOCTOR_WHATSAPP = {
    107: '+918861839506',  # Dr. Shyam
    108: '+919731014619',  # Dr. Kaushik
    109: '+918310117314',  # Dr. Paavani
    110: '+919645380381'   # Dr. Deepa
}

# Alternative: Twilio Configuration (if you want to use Twilio later)
# TWILIO_ACCOUNT_SID = "your_account_sid_here"
# TWILIO_AUTH_TOKEN = "your_auth_token_here"
# TWILIO_WHATSAPP_NUMBER = "whatsapp:+14155238886"

# Usage Notes:
# - Emails will only be sent for doctors with IDs 107-110
# - Other doctors (demo doctors) will work without email functionality
# - Make sure to update SENDER_EMAIL and SENDER_PASSWORD above
# - For WhatsApp: Update TWILIO credentials and doctor phone numbers
# - After updating, restart the backend server
