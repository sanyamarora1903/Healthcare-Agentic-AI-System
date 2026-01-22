from agent import healthcare_agent
from memory.qdrant_store import store_fact

def main():
    print("\n--- Healthcare Memory Agent ---")
    print("1. Store medical fact")
    print("2. Ask medical question")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        fact = input("Enter medical fact: ")
        fact_type = input("Enter fact type (diagnosis/allergy/medication): ")
        store_fact(fact, fact_type)
        print("✅ Fact stored successfully")

    elif choice == "2":
        symptoms = input("Enter symptoms (comma-separated): ").split(",")
        symptoms = [s.strip().lower() for s in symptoms]

        age = input("Enter age: ")
        gender = input("Enter gender: ")

        patient_data = {
            "symptoms": symptoms,
            "age": age,
            "gender": gender
        }

        print("\nThinking...\n")
        answer = healthcare_agent(patient_data)
        print(answer)

    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()
