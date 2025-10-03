#!/usr/bin/env python3
"""
Database migration script to add WhatsApp fields to UserPreferences
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.getcwd())

def migrate_database():
    """Add WhatsApp fields to existing UserPreferences records"""
    try:
        from app import app, db, UserPreferences
        
        with app.app_context():
            print("🔧 Checking database schema...")
            
            # Check if the table exists and create all tables
            db.create_all()
            print("✅ Database tables created/updated")
            
            # Check existing UserPreferences records
            prefs = UserPreferences.query.all()
            print(f"📊 Found {len(prefs)} existing user preference records")
            
            # Update existing records to have default WhatsApp values
            updated_count = 0
            for pref in prefs:
                if not hasattr(pref, 'whatsapp_enabled') or pref.whatsapp_enabled is None:
                    pref.whatsapp_enabled = False
                    updated_count += 1
                
                if not hasattr(pref, 'whatsapp_phone') or pref.whatsapp_phone is None:
                    pref.whatsapp_phone = ""
                    updated_count += 1
            
            if updated_count > 0:
                db.session.commit()
                print(f"✅ Updated {updated_count} user preference records with WhatsApp defaults")
            else:
                print("✅ All user preferences already have WhatsApp fields")
            
            # Test creating a new UserPreferences object
            test_pref = UserPreferences(user_id=999)  # Temporary test
            print(f"✅ WhatsApp fields available:")
            print(f"   - whatsapp_enabled: {hasattr(test_pref, 'whatsapp_enabled')}")
            print(f"   - whatsapp_phone: {hasattr(test_pref, 'whatsapp_phone')}")
            
            return True
            
    except Exception as e:
        print(f"❌ Database migration error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prescription_extraction():
    """Test if the prescription extraction function works"""
    try:
        from app import extract_medicine_data
        
        print("\n🔬 Testing prescription text extraction...")
        
        # Test with sample prescription text
        test_text = """
        Tab Paracetamol 500mg 2x 7 days
        Cap Amoxicillin 250mg 3x 5 days
        Syrup Crocin 1x 3 days
        """
        
        result = extract_medicine_data(test_text)
        print(f"✅ Extracted {len(result)} medicines:")
        for i, med in enumerate(result, 1):
            print(f"   {i}. {med}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prescription extraction error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🏥 MediReminder Database Migration & Test")
    print("=" * 50)
    
    # Run database migration
    db_success = migrate_database()
    
    # Test prescription extraction
    extraction_success = test_prescription_extraction()
    
    print("\n" + "=" * 50)
    if db_success and extraction_success:
        print("🎉 All tests passed! The application should work correctly.")
    else:
        print("❌ Some issues found. Please check the errors above.")
    
    print("\n💡 Next steps:")
    print("   1. Start the Flask app: python app.py")
    print("   2. Go to Profile Settings")
    print("   3. Configure WhatsApp notifications")
    print("   4. Test prescription upload")
    print("   5. Create reminders and test notifications")