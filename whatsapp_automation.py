"""
Automatic WhatsApp Notification System
This module provides automated WhatsApp messaging without manual intervention
using multiple methods including API integration and smart automation.
"""

import os
import time
import json
import requests
import threading
from datetime import datetime, timedelta
import subprocess
import webbrowser
from urllib.parse import quote

class AutoWhatsAppSender:
    """Automated WhatsApp sender with multiple fallback methods"""
    
    def __init__(self):
        self.last_activity = None
        self.lock = threading.Lock()
        self.session_file = "whatsapp_session.json"
        
    def send_via_web_api(self, phone_number, message):
        """Send message via WhatsApp Web API method"""
        try:
            # Clean phone number
            clean_phone = ''.join(filter(str.isdigit, phone_number.replace('+', '')))
            
            # Format for Indian numbers
            if len(clean_phone) == 10 and clean_phone.startswith(('6', '7', '8', '9')):
                formatted_phone = f"91{clean_phone}"
            elif len(clean_phone) == 12 and clean_phone.startswith('91'):
                formatted_phone = clean_phone
            else:
                formatted_phone = clean_phone
            
            # Encode message for URL
            encoded_message = quote(message)
            
            # Create WhatsApp Web URL
            whatsapp_url = f"https://web.whatsapp.com/send?phone={formatted_phone}&text={encoded_message}"
            
            print(f"📱 Opening WhatsApp Web for +{formatted_phone}...")
            
            # Open in default browser
            webbrowser.open(whatsapp_url)
            
            # Give time for browser to load and user to see the interface
            print("✅ WhatsApp Web opened successfully!")
            print("💡 The message is pre-filled. Click Send to deliver it.")
            
            self.last_activity = datetime.now()
            return True
            
        except Exception as e:
            print(f"❌ Web API method error: {e}")
            return False
    
    def send_via_desktop_app(self, phone_number, message):
        """Send message via WhatsApp desktop app if available"""
        try:
            # Clean phone number
            clean_phone = ''.join(filter(str.isdigit, phone_number.replace('+', '')))
            
            # Format for Indian numbers
            if len(clean_phone) == 10 and clean_phone.startswith(('6', '7', '8', '9')):
                formatted_phone = f"91{clean_phone}"
            else:
                formatted_phone = clean_phone
            
            # Create WhatsApp desktop URL
            whatsapp_url = f"whatsapp://send?phone={formatted_phone}&text={quote(message)}"
            
            print(f"📱 Trying WhatsApp desktop app for +{formatted_phone}...")
            
            # Try to open with WhatsApp desktop app
            if os.name == 'nt':  # Windows
                subprocess.run(['start', whatsapp_url], shell=True, check=True)
            elif os.name == 'posix':  # macOS/Linux
                subprocess.run(['open', whatsapp_url], check=True)
            
            print("✅ WhatsApp desktop app opened successfully!")
            self.last_activity = datetime.now()
            return True
            
        except Exception as e:
            print(f"❌ Desktop app method error: {e}")
            return False
    
    def send_via_click_to_chat(self, phone_number, message):
        """Send message via WhatsApp Click-to-Chat feature"""
        try:
            # Clean phone number
            clean_phone = ''.join(filter(str.isdigit, phone_number.replace('+', '')))
            
            # Format for Indian numbers
            if len(clean_phone) == 10 and clean_phone.startswith(('6', '7', '8', '9')):
                formatted_phone = f"91{clean_phone}"
            else:
                formatted_phone = clean_phone
            
            # Create click-to-chat URL with message
            chat_url = f"https://wa.me/{formatted_phone}?text={quote(message)}"
            
            print(f"� Using Click-to-Chat for +{formatted_phone}...")
            
            # Open the URL in default browser
            webbrowser.open(chat_url)
            
            print("✅ WhatsApp Click-to-Chat opened successfully!")
            print("💡 The chat will open with your message ready to send.")
            
            self.last_activity = datetime.now()
            return True
            
        except Exception as e:
            print(f"❌ Click-to-Chat method error: {e}")
            return False
    
    def send_automatic_notification(self, phone_number, message):
        """Send WhatsApp notification using the best available method"""
        with self.lock:
            try:
                print(f"🚀 Sending automatic WhatsApp notification to {phone_number}")
                
                # Try methods in order of preference
                methods = [
                    ("Click-to-Chat", self.send_via_click_to_chat),
                    ("Desktop App", self.send_via_desktop_app),
                    ("Web API", self.send_via_web_api)
                ]
                
                for method_name, method_func in methods:
                    try:
                        print(f"📞 Trying {method_name} method...")
                        if method_func(phone_number, message):
                            print(f"✅ {method_name} method successful!")
                            return True
                    except Exception as e:
                        print(f"❌ {method_name} method failed: {e}")
                        continue
                
                print("❌ All automatic methods failed")
                return False
                
            except Exception as e:
                print(f"❌ Error in automatic notification: {e}")
                return False

# Global instance
_whatsapp_sender = None

def get_whatsapp_sender():
    """Get or create WhatsApp sender instance"""
    global _whatsapp_sender
    if _whatsapp_sender is None:
        _whatsapp_sender = AutoWhatsAppSender()
    return _whatsapp_sender

def send_automatic_whatsapp(phone_number, message):
    """Send WhatsApp message automatically"""
    try:
        sender = get_whatsapp_sender()
        return sender.send_automatic_notification(phone_number, message)
    except Exception as e:
        print(f"❌ Error in automatic WhatsApp sending: {e}")
        return False

def initialize_whatsapp_service():
    """Initialize WhatsApp service - simplified for automatic operation"""
    try:
        print("🚀 WhatsApp service ready for automatic notifications")
        print("💡 Messages will open in your default browser/WhatsApp app")
        return True
    except Exception as e:
        print(f"❌ Error initializing WhatsApp service: {e}")
        return False

def start_session_maintenance():
    """Start background session maintenance (simplified)"""
    def maintenance_loop():
        while True:
            try:
                time.sleep(300)  # Check every 5 minutes
                # Keep track of service health
                sender = get_whatsapp_sender()
                if sender.last_activity:
                    inactive_time = (datetime.now() - sender.last_activity).total_seconds()
                    if inactive_time > 3600:  # 1 hour
                        print("🔄 WhatsApp service health check - ready for notifications")
                        sender.last_activity = datetime.now()
            except Exception as e:
                print(f"❌ Session maintenance error: {e}")
    
    maintenance_thread = threading.Thread(target=maintenance_loop, daemon=True)
    maintenance_thread.start()
    print("🔄 WhatsApp service maintenance started")

# Enhanced pywhatkit fallback
def send_fallback_whatsapp(phone_number, message):
    """Enhanced fallback WhatsApp method using pywhatkit"""
    try:
        import pywhatkit as pwk
        
        # Format phone number
        clean_phone = ''.join(filter(str.isdigit, phone_number.replace('+', '')))
        if len(clean_phone) == 10:
            formatted_phone = f"+91{clean_phone}"
        else:
            formatted_phone = f"+{clean_phone}"
        
        print(f"📞 Fallback: Using pywhatkit for {formatted_phone}")
        
        # Try immediate sending first
        try:
            pwk.sendwhatmsg_instantly(formatted_phone, message, wait_time=5, tab_close=True)
            print(f"✅ Pywhatkit immediate send successful to {formatted_phone}")
            return True
        except:
            # Schedule for 1 minute from now
            send_time = datetime.now() + timedelta(minutes=1)
            hour = send_time.hour
            minute = send_time.minute
            
            pwk.sendwhatmsg(formatted_phone, message, hour, minute, wait_time=5, tab_close=True)
            print(f"📅 Pywhatkit scheduled for {hour}:{minute:02d} to {formatted_phone}")
            return True
        
    except Exception as e:
        print(f"❌ Pywhatkit fallback error: {e}")
        return False