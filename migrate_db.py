#!/usr/bin/env python3
"""
Database migration script to add WhatsApp fields to UserPreferences table
"""

import sqlite3
import os

def migrate_database():
    """Add WhatsApp columns to the existing database"""
    db_path = "instance/medicine_reminder.db"
    
    try:
        # Create instance directory if it doesn't exist
        os.makedirs("instance", exist_ok=True)
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("🔧 Checking existing database schema...")
        
        # Check if WhatsApp columns already exist
        cursor.execute("PRAGMA table_info(user_preferences)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        print(f"📊 Current columns in user_preferences: {column_names}")
        
        # Add WhatsApp columns if they don't exist
        if 'whatsapp_enabled' not in column_names:
            print("➕ Adding whatsapp_enabled column...")
            cursor.execute("ALTER TABLE user_preferences ADD COLUMN whatsapp_enabled BOOLEAN DEFAULT 0")
            print("✅ Added whatsapp_enabled column")
        else:
            print("✅ whatsapp_enabled column already exists")
        
        if 'whatsapp_phone' not in column_names:
            print("➕ Adding whatsapp_phone column...")
            cursor.execute("ALTER TABLE user_preferences ADD COLUMN whatsapp_phone VARCHAR(20) DEFAULT ''")
            print("✅ Added whatsapp_phone column")
        else:
            print("✅ whatsapp_phone column already exists")
        
        # Commit changes
        conn.commit()
        
        # Verify the changes
        cursor.execute("PRAGMA table_info(user_preferences)")
        updated_columns = cursor.fetchall()
        updated_column_names = [col[1] for col in updated_columns]
        
        print(f"📊 Updated columns in user_preferences: {updated_column_names}")
        
        # Close connection
        conn.close()
        
        print("🎉 Database migration completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database migration failed: {e}")
        return False

def create_sample_data():
    """Create sample data for testing"""
    try:
        import sys
        import os
        sys.path.insert(0, os.getcwd())
        
        from app import app, db, User, UserPreferences
        
        with app.app_context():
            # Check if we have any users
            user_count = User.query.count()
            print(f"📊 Found {user_count} users in the database")
            
            if user_count == 0:
                print("🔧 Creating a test user...")
                test_user = User(
                    username="testuser",
                    email="test@example.com",
                    password_hash="test_hash"  # In real app, this would be properly hashed
                )
                db.session.add(test_user)
                db.session.commit()
                print("✅ Created test user")
                
                # Create preferences for the test user
                test_prefs = UserPreferences(user_id=test_user.id)
                db.session.add(test_prefs)
                db.session.commit()
                print("✅ Created test user preferences")
            
        return True
        
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        return False

if __name__ == "__main__":
    print("🏥 MediReminder Database Migration")
    print("=" * 40)
    
    # Run database migration
    migration_success = migrate_database()
    
    if migration_success:
        print("\n🔬 Testing with Flask app...")
        sample_success = create_sample_data()
    
    print("\n" + "=" * 40)
    if migration_success:
        print("✅ Database migration completed!")
        print("\n💡 You can now:")
        print("   1. Start the Flask app: python app.py")
        print("   2. Register/login to the application")
        print("   3. Go to Profile Settings")
        print("   4. Configure WhatsApp notifications")
        print("   5. Test prescription upload and reminders")
    else:
        print("❌ Migration failed. Please check the errors above.")