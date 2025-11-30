"""
Student database management using SQLite
"""
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional


class StudentDatabase:
    def __init__(self, db_path: str = 'database/students.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Students table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone TEXT UNIQUE NOT NULL,
                name TEXT,
                grade INTEGER,
                target_schools TEXT,
                interests TEXT,
                current_focus TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                category TEXT,
                summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students (id)
            )
        ''')
        
        # Messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                sender TEXT,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"[OK] Database initialized at {self.db_path}")
    
    def get_or_create_student(self, phone: str, name: str = None) -> int:
        """Get existing student or create new one"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if student exists
        cursor.execute('SELECT id FROM students WHERE phone = ?', (phone,))
        result = cursor.fetchone()
        
        if result:
            student_id = result[0]
        else:
            # Create new student
            cursor.execute(
                'INSERT INTO students (phone, name) VALUES (?, ?)',
                (phone, name)
            )
            student_id = cursor.lastrowid
            print(f"Created new student: {name or phone}")
        
        conn.commit()
        conn.close()
        return student_id
    
    def update_student_profile(self, phone: str, **kwargs):
        """Update student profile fields"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get student_id (provide dummy name if needed)
        cursor.execute('SELECT id FROM students WHERE phone = ?', (phone,))
        result = cursor.fetchone()
        if result:
            student_id = result[0]
        else:
            student_id = self.get_or_create_student(phone, "Student")
        
        # Build update query
        valid_fields = ['name', 'grade', 'target_schools', 'interests', 'current_focus']
        updates = []
        values = []
        
        for field, value in kwargs.items():
            if field in valid_fields and value is not None:
                updates.append(f"{field} = ?")
                values.append(value)
        
        if updates:
            values.append(datetime.now())
            values.append(student_id)
            query = f"UPDATE students SET {', '.join(updates)}, updated_at = ? WHERE id = ?"
            cursor.execute(query, values)
            conn.commit()
        
        conn.close()
    
    def get_student_context(self, phone: str) -> Dict:
        """Get student profile and recent conversation history"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get student profile
        cursor.execute('SELECT * FROM students WHERE phone = ?', (phone,))
        student = cursor.fetchone()
        
        if not student:
            conn.close()
            return None
        
        student_dict = dict(student)
        
        # Get recent conversations (last 5)
        cursor.execute('''
            SELECT c.category, c.summary, c.created_at,
                   GROUP_CONCAT(m.content, ' | ') as messages
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            WHERE c.student_id = ?
            GROUP BY c.id
            ORDER BY c.created_at DESC
            LIMIT 5
        ''', (student_dict['id'],))
        
        conversations = [dict(row) for row in cursor.fetchall()]
        student_dict['recent_conversations'] = conversations
        
        conn.close()
        return student_dict
    
    def create_conversation(self, student_id: int, category: str, summary: str = None) -> int:
        """Create a new conversation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO conversations (student_id, category, summary) VALUES (?, ?, ?)',
            (student_id, category, summary)
        )
        conversation_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        return conversation_id
    
    def add_message(self, conversation_id: int, sender: str, content: str):
        """Add a message to a conversation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO messages (conversation_id, sender, content) VALUES (?, ?, ?)',
            (conversation_id, sender, content)
        )
        
        conn.commit()
        conn.close()
    
    def get_conversation_history(self, phone: str, limit: int = 10) -> List[Dict]:
        """Get recent message history for a student"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT m.sender, m.content, m.timestamp
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            JOIN students s ON c.student_id = s.id
            WHERE s.phone = ?
            ORDER BY m.timestamp DESC
            LIMIT ?
        ''', (phone, limit))
        
        messages = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return list(reversed(messages))  # Return in chronological order
    
    def populate_from_processed_data(self, processed_data_path: str = 'data/processed_chats.json'):
        """Populate database from processed chat data"""
        with open(processed_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Populating database with {len(data)} Q&A pairs...")
        
        for idx, qa in enumerate(data):
            # Get or create student
            student_id = self.get_or_create_student(
                qa.get('student_phone', f'+phone_{idx}'),
                qa.get('student_name', 'Student')
            )
            
            # Create conversation
            conv_id = self.create_conversation(
                student_id,
                qa.get('category', 'general_inquiry'),
                qa.get('question', '')[:100]  # Use first 100 chars as summary
            )
            
            # Add messages
            self.add_message(conv_id, 'student', qa.get('question', ''))
            self.add_message(conv_id, 'counselor', qa.get('answer', ''))
            
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1} conversations...")
        
        print(f"[OK] Database populated successfully!")


if __name__ == "__main__":
    # Initialize database
    db = StudentDatabase()
    
    # Test if processed data exists
    import os
    if os.path.exists('data/processed_chats.json'):
        db.populate_from_processed_data('data/processed_chats.json')
        
        # Test retrieval
        print("\n--- Testing Database ---")
        test_phone = "+1234567890"
        context = db.get_student_context(test_phone)
        if context:
            print(f"\nStudent: {context['name']}")
            print(f"Recent conversations: {len(context['recent_conversations'])}")
    else:
        print("No processed data found. Run data processing first.")