import os
from typing import Dict, List, Optional
from api_client import llm_client
from embeddings_store import EmbeddingsStore
from database import StudentDatabase

class StudentBot:
    def __init__(self):
        self.embeddings_store = EmbeddingsStore()
        self.db = StudentDatabase()
        
        # Load embeddings
        try:
            self.embeddings_store.load('data/embeddings.pkl')
            print("[OK] Bot initialized with embeddings")
        except FileNotFoundError:
            print("⚠️ No embeddings found. Run Day 2 setup first.")
    
    def categorize_question(self, question: str) -> str:
        """Quick categorization of incoming question"""
        try:
            response = llm_client.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": """Categorize this question into ONE category:
                        college_applications, internships, research_opportunities, essay_writing,
                        standardized_tests, course_selection, extracurriculars, career_guidance, general_inquiry
                        
                        Respond with ONLY the category name."""
                    },
                    {"role": "user", "content": question}
                ],
                temperature=0.3,
                max_tokens=20
            )
            return response.strip().lower()
        except:
            return "general_inquiry"
    
    def build_context(self, student_phone: str, question: str) -> Dict:
        """Build context from student profile and similar conversations"""
        # Get student context
        student_context = self.db.get_student_context(student_phone)
        
        # Get conversation history
        conversation_history = self.db.get_conversation_history(student_phone, limit=6)
        
        # Categorize current question
        category = self.categorize_question(question)
        
        # Search for similar Q&A pairs
        similar_pairs = self.embeddings_store.search_similar(
            question, 
            top_k=3,
            category_filter=category
        )
        
        return {
            'student': student_context,
            'conversation_history': conversation_history,
            'category': category,
            'similar_examples': similar_pairs
        }
    
    def generate_response(self, question: str, context: Dict) -> str:
        """Generate human-like response using LLM with context"""
        
        # Build system prompt with context
        system_prompt = self._build_system_prompt(context)
        
        # Build conversation history for model
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent conversation history (more context)
        if context['conversation_history']:
            for msg in context['conversation_history'][-8:]:  # Last 8 messages for better memory
                role = "user" if msg['sender'] == 'student' else "assistant"
                messages.append({"role": role, "content": msg['content']})
        
        # Add current question
        messages.append({"role": "user", "content": question})
        
        try:
            response = llm_client.chat_completion(
                messages=messages,
                temperature=0.8,  # More natural/varied responses
                max_tokens=150  # Shorter responses (was 500)
            )
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Having trouble right now. Want me to connect you with a counselor?"
    
    def _build_system_prompt(self, context: Dict) -> str:
        """Build system prompt with all relevant context"""
        
        student = context.get('student')
        category = context.get('category', 'general')
        similar_examples = context.get('similar_examples', [])
        
        # Base prompt
        prompt = """You are a friendly student counselor - like a helpful friend who knows a lot about college stuff.

PERSONALITY:
- Warm, supportive, conversational - like texting a friend
- Use casual language: "Sounds good!", "Makes sense!", "Awesome!"
- Add personality: "Oh nice!", "I feel you", "That's exciting!"
- Don't be robotic or overly formal
- Match their vibe - if casual, be casual; if serious, be focused

RESPONSE STYLE:
- KEEP IT SHORT: 1-2 sentences unless they ask for details
- Natural conversation flow - don't always end with questions
- If something is done/answered, just acknowledge and wait for them
- Examples:
  ✓ "Sounds good! 3:30 Monday works - I'll have my team confirm."
  ✗ "Monday at 3:30 is set. What would you like to discuss in this meeting?"
  
  ✓ "Got it - History and Bio! I'll prep some options for you."
  ✗ "History and Biology are great choices. What are your GPA and test scores?"

WHEN TO STOP:
- If they've given you info (time, major, etc.) → acknowledge and stop
- Don't keep digging unless they seem to want more help
- Let THEM drive - respond to what they ask
- If conversation feels done, just say something friendly and wait

SPECIAL CASES:
- Scheduling meetings → Get time/date, then generate calendar link
- Materials/resources → Ask what specifically, then provide or connect to human
- Just chatting → Be friendly, don't force business talk

"""
        
        # Add student context if available
        if student:
            prompt += f"\nSTUDENT CONTEXT:\n"
            if student.get('name'):
                prompt += f"- Name: {student['name']}\n"
            if student.get('grade'):
                prompt += f"- Grade: {student['grade']}\n"
            if student.get('interests'):
                prompt += f"- Interests: {student['interests']}\n"
            if student.get('target_schools'):
                prompt += f"- Target Schools: {student['target_schools']}\n"
            if student.get('current_focus'):
                prompt += f"- Current Focus: {student['current_focus']}\n"
        
        # Add similar examples for style reference
        if similar_examples:
            prompt += "\n\nREFERENCE EXAMPLES (for style and content inspiration):\n"
            for idx, (qa, score) in enumerate(similar_examples[:2], 1):
                prompt += f"\nExample {idx}:\n"
                prompt += f"Q: {qa['question']}\n"
                prompt += f"A: {qa['answer']}\n"
        
        prompt += """\n
Remember: Be helpful, specific, and authentic. If you don't know something, be honest and offer to connect 
them with someone who can help. Your goal is to guide them toward their academic and career success."""
        
        return prompt
    
    def handle_message(self, student_phone: str, student_name: str, message: str) -> str:
        """Main entry point for handling incoming messages"""
        
        # Ensure student exists in database
        student_id = self.db.get_or_create_student(student_phone, student_name)
        
        # Build context
        context = self.build_context(student_phone, message)
        
        # Check if this is a meeting request
        meeting_response = self._handle_meeting_request(student_phone, message, context)
        if meeting_response:
            return meeting_response
        
        # Extract and update student info from conversation
        self._extract_student_info(student_phone, message, context)
        
        # Generate response
        response = self.generate_response(message, context)
        
        # Log conversation
        conv_id = self.db.create_conversation(
            student_id, 
            context['category'],
            message[:100]
        )
        self.db.add_message(conv_id, 'student', message)
        self.db.add_message(conv_id, 'bot', response)
        
        return response
    
    def _handle_meeting_request(self, student_phone: str, message: str, context: Dict) -> Optional[str]:
        """Handle meeting scheduling requests"""
        message_lower = message.lower()
        
        # Check if message contains meeting-related keywords
        meeting_keywords = ['meeting', 'schedule', 'appointment', 'call', 'meet', 'talk to counselor']
        has_meeting_request = any(keyword in message_lower for keyword in meeting_keywords)
        
        if not has_meeting_request:
            return None
        
        # Check conversation history for collected info
        history = self.db.get_conversation_history(student_phone, limit=10)
        
        # Extract day/time from recent messages
        import re
        from datetime import datetime, timedelta
        
        days = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6}
        time_pattern = r'\b(\d{1,2}):?(\d{2})?\s*(am|pm)?\b'
        
        collected_day = None
        collected_time = None
        collected_topic = None
        
        # Check recent messages
        for msg in history[-6:]:
            msg_lower = msg['content'].lower()
            
            # Check for day
            for day_name, day_num in days.items():
                if day_name in msg_lower:
                    collected_day = day_name.title()
                    break
            
            # Check for time
            time_match = re.search(time_pattern, msg_lower)
            if time_match:
                hour = time_match.group(1)
                minute = time_match.group(2) or '00'
                period = time_match.group(3) or ''
                collected_time = f"{hour}:{minute} {period}".strip()
            
            # Check for topic mentions
            if any(word in msg_lower for word in ['discuss', 'about', 'help with', 'talk about']):
                # Extract topic context
                topic_keywords = ['college', 'essay', 'application', 'uni', 'university', 'major', 'internship']
                for keyword in topic_keywords:
                    if keyword in msg_lower:
                        collected_topic = keyword
                        break
        
        # Generate meeting link if we have enough info
        if collected_day and collected_time:
            # Create Calendly-style link (dummy for now)
            meeting_link = f"https://calendly.com/counselor/30min?date={collected_day.lower()}&time={collected_time.replace(':', '').replace(' ', '')}"
            
            topic_text = f" about {collected_topic}" if collected_topic else ""
            
            return f"Perfect! I've set up a meeting for {collected_day} at {collected_time}{topic_text}. Here's your link to confirm: {meeting_link}\n\nYou'll get a reminder before the meeting. See you then! 😊"
        
        return None
    
    def _extract_student_info(self, student_phone: str, message: str, context: Dict):
        """Extract student info (major, interests) from conversation"""
        message_lower = message.lower()
        
        # Check for major mentions
        majors = ['economics', 'finance', 'computer science', 'cs', 'biology', 'engineering', 
                  'psychology', 'political science', 'mathematics', 'physics', 'chemistry',
                  'business', 'pre-med', 'premed', 'law', 'english', 'history']
        
        for major in majors:
            if major in message_lower:
                # Update student profile
                current_context = self.db.get_student_context(student_phone)
                if current_context and not current_context.get('interests'):
                    self.db.update_student_profile(
                        student_phone,
                        interests=major.title(),
                        current_focus=context.get('category', 'general')
                    )
                break
    
    def check_handoff_needed(self, question: str, response: str) -> bool:
        """Determine if question should be handed off to human counselor"""
        
        # Keywords that might indicate complex situations
        handoff_keywords = [
            'emergency', 'crisis', 'suicide', 'depressed', 'harm',
            'legal', 'financial aid appeal', 'scholarship negotiation',
            'urgent deadline', 'specific school policy'
        ]
        
        question_lower = question.lower()
        
        for keyword in handoff_keywords:
            if keyword in question_lower:
                return True
        
        # Don't flag short responses as needing handoff
        return False


if __name__ == "__main__":
    # Test the bot
    bot = StudentBot()
    
    print("\n🤖 Testing Student Bot\n")
    print("=" * 60)
    
    # Test question
    test_question = "I'm interested in finding computer science research opportunities. I'm a junior and really passionate about AI."
    test_phone = "+1234567890"
    test_name = "Test Student"
    
    print(f"Question: {test_question}\n")
    
    response = bot.handle_message(test_phone, test_name, test_question)
    
    print(f"Bot Response:\n{response}")
    print("\n" + "=" * 60)
    
    # Check if handoff needed
    needs_handoff = bot.check_handoff_needed(test_question, response)
    print(f"\nNeeds human counselor: {needs_handoff}")