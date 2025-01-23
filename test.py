import logging
from neu import (
    load_model_with_questions_and_answers,
    simulate_question_answering,
    find_similar_question
)

# Deaktiviere alle Logging-Ausgaben außer kritischen Fehlern
logging.getLogger().setLevel(logging.CRITICAL)

# Modell laden
category_nodes, questions = load_model_with_questions_and_answers("model_with_qa.json")

def test_model_with_answers(category_nodes, questions, query):
    """
    Testet das Modell mit einer Abfrage und gibt die gefundene Frage, Antwort und Gewichtung aus.

    Args:
        category_nodes (list): Liste der Kategorie-Knoten.
        questions (list): Liste der Fragen.
        query (str): Die Abfrage, nach der gesucht werden soll.
    """
    # Suche nach der ähnlichsten Frage im Modell
    matched_question = find_similar_question(questions, query)

    if matched_question and matched_question.get('question') != "Keine passende Frage gefunden":
        answer = matched_question.get('answer', 'Keine Antwort verfügbar')

        # Simulation der Fragebeantwortung (Gewichtung/Aktivierung)
        activation = simulate_question_answering(category_nodes, matched_question['question'], questions)

        # Nur relevante Informationen ausgeben
        print(f"Frage: \"{query}\"")
        print(f"Antwort: \"{answer}\"")
        print(f"Gewichtung: {activation:.2f}\n")

    else:
        # Falls keine passende Frage gefunden wurde
        print(f"Frage: \"{query}\"")
        print("Antwort: \"Keine passende Frage gefunden\"")
        print("Gewichtung: 0.00\n")

# Testaufrufe mit den gegebenen Fragen
if category_nodes and questions:
    test_model_with_answers(category_nodes, questions, "Bist du dabei?")
    test_model_with_answers(category_nodes, questions, "Du Idiot!")
    test_model_with_answers(category_nodes, questions, "Frag Tom!")
    test_model_with_answers(category_nodes, questions, "Wer hat gesprochen?")
    test_model_with_answers(category_nodes, questions, "Prüfen Sie das nach!")
    test_model_with_answers(category_nodes, questions, "Komm schnell!")

    test_model_with_answers(category_nodes, questions, "Wer geht?")
    test_model_with_answers(category_nodes, questions, "Wen kümmert das schon?")
    test_model_with_answers(category_nodes, questions, "Wir haben gewonnen!")
    test_model_with_answers(category_nodes, questions, "Wir lächelten.")
    test_model_with_answers(category_nodes, questions, "Versuch es noch einmal.")
    test_model_with_answers(category_nodes, questions, "Tom ist schüchtern.")
else:
    logging.critical("Kein Modell gefunden.")
