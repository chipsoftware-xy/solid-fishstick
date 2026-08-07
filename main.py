from app.llm.chat_session import ChatSession

chat = ChatSession()

print("Spark is ready.")

while True:

    user = input("\nYou: ")

    if user.lower() == "exit":
        break

    reply = chat.send(user)

    print(f"\nSpark: {reply}")