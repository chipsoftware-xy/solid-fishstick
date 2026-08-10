from app.llm.ollama_client import LLM
from app.llm.prompts import SYSTEM_PROMPT, OPENING_MESSAGE
from app.llm.knowledge_prompts import EMPLOYEE_ID_PROMPT
from app.knowledge_base import KnowledgeBase

REJECTION_PHRASES = {
    "no",
    "nope",
    "not interested",
    "not really",
    "no thanks",
}

BUSY_PHRASES = {
    "i am busy",
    "i'm busy",
    "i am busy right now",
    "i'm busy right now",
    "busy",
}
SPECIALIST_PHRASES = {
    "connect you with one of our specialists",
    "connect you with a specialist",
    "connect you to a specialist",
    "speak with one of our specialists",
    "speak to one of our specialists",
    "talk to one of our specialists",
    "speak with a specialist",
    "talk to a specialist",
    "specialist to discuss",
    "specialist call",
    "schedule a call",
    "arrange a call"
}

SPECIALIST_YES_PHRASES = {
    "yes",
    "yeah",
    "yep",
    "sure",
    "okay",
    "ok",
    "fine",
    "alright",
    "yes sure",
    "yeah sure",
    "yes please",
    "sure thing",
    "absolutely",
    "of course",
}
SCHEDULING_PHRASES = {
    "call me tomorrow",
    "call tomorrow",
    "contact me tomorrow",
    "tomorrow is fine",
    "tomorrow works",
    "call me later",
    "contact me later",
    "call me another time",
}


class ChatSession:

    def __init__(self):

        self.waiting_for_callback_confirmation = False

        self.llm = LLM()
        self.knowledge = KnowledgeBase()
        self.call_started = False
        self.waiting_for_callback_time = False
        self.waiting_for_specialist_details = False
        from app.llm.knowledge_prompts import EMPLOYEE_ID_PROMPT

        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

    def send(self, user_message):

        user_message = user_message.strip()
        clean = user_message.lower()

        # Empty input
        if not user_message:
            return "Sorry, I didn't catch that."
        #Opening Message
        if not self.call_started:
            self.call_started = True
            return OPENING_MESSAGE
        if any(phrase in clean for phrase in {
            "what is this about",
            "what's this about",
            "what is this regarding",
            "what's this regarding",
            "why are you calling",
            "what do you offer",
            "what do you guys do",
            "what is smart chip"}):
            return (
                "We provide solutions for employee attendance, access control, "
                "RFID tracking, ID cards and customized software and hardware integration. "
                "For a limited time, we are also offering complimentary services for "
                "ID card printer servicing, card designing and related software services. "
                "I wanted to see if any of these areas are relevant to your business."
                )
        # Rejection
        if clean in REJECTION_PHRASES:
            return (
                "No problem at all."
                "Thanks for your time. Have a great day."
                            )

        # Busy
        if clean in BUSY_PHRASES or "busy" in clean:
            self.waiting_for_callback_confirmation = True

            return "No problem. Would another time be better?"

        # Callback response
        if self.waiting_for_callback_confirmation:

            if clean in {
                "yes",
                "yeah",
                "sure",
                "yep",
                "fine",
                "okay",
                "yeah sure",
                "alright",
                "yeah okay",
                "sure alright"
            }:
                self.waiting_for_callback_confirmation = False
                self.waiting_for_callback_time = True

                return "Okay, what time would be suitable for you?"

            if clean in {
                "no",
                "nope",
                "not really",
                "no thanks"
            }:
                self.waiting_for_callback_confirmation = False

                return (
                    "No problem, thank you for your time!"
                )
        # Customer directly requests a callback
        if clean in SCHEDULING_PHRASES:
            self.waiting_for_callback_time = True
            return "Certainly. What time would be convenient for you?"
        
        #callback time
        if self.waiting_for_callback_time:
            self.waiting_for_callback_time = False
            return (f"Perfect, our specialist will contact you at {user_message}. "
                    "Thank you for your time, have a great day!")
        
        #Add customer message to conversation memory
        self.messages.append(
            {
                "role": "user",
                "content": user_message
            }
        )

        # Search company knowledge
        knowledge_results = self.knowledge.search(
            user_message,
            n_results=3
        )

        knowledge_context = "\n\n".join(
            knowledge_results
        )

        # Create the LLM messages with retrieved knowledge
        llm_messages = [
            self.messages[0],
            {
        "role": "system",
        "content": f"""
        {EMPLOYEE_ID_PROMPT}
        Here is verified information about Smart Chip Tree that may be
        relevant to the customer's current message.
        Use this information when it is relevant.
        Do not mention that you were given this information.
        Do not dump the information on the customer.
        Do not invent information beyond it.
        SMART CHIP TREE KNOWLEDGE:
        {knowledge_context}
        CONVERSATION GUIDANCE:
        You are on a sales call.
        Your job is to understand the customer's need, explain how Smart Chip
        Tree could potentially help, and move toward a useful next step.
        Do not turn the conversation into an interview.
        If the customer has already given enough information, stop asking
        questions and provide a relevant response or solution.
        If the customer asks whether Smart Chip Tree can provide something
        that is related to its technology domain, you may describe it as a
        customized solution when appropriate.
        If the customer asks about something outside the company's domain,
        do not invent a solution for it.
        If the customer asks an unrelated casual question, answer very briefly
        if appropriate and naturally return to the business conversation.
        Do not ask a question simply because you need to keep the conversation going.
        Do not repeatedly ask what the customer's business does.
        Do not repeat questions that have already been answered."""
     },
    *self.messages[1:]
    ]
        
        # Generate response
        reply = self.llm.generate(
            llm_messages
        )
        #Detect when Spark is offering a specialist call
        if any(phrase in reply.lower() for phrase in SPECIALIST_PHRASES):
            self.waiting_for_callback_confirmation = True

        # Store assistant response
        self.messages.append(
            {
                "role": "assistant",
                "content": reply
            }
        )


        # Keep conversation memory small
        if len(self.messages) > 7:
            self.messages = [
                self.messages[0],
                *self.messages[-6:]
            ]

        return reply