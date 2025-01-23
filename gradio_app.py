import logging
import gradio as gr
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
    Testet das Modell mit einer Abfrage und gibt die gefundene Frage, Antwort und Gewichtung zurück.

    Args:
        category_nodes (list): Liste der Kategorie-Knoten.
        questions (list): Liste der Fragen.
        query (str): Die Abfrage, nach der gesucht werden soll.

    Returns:
        tuple: Die gefundene Frage, Antwort und Gewichtung.
    """
    # Suche nach der ähnlichsten Frage im Modell
    matched_question = find_similar_question(questions, query)

    if matched_question and matched_question.get('question') != "Keine passende Frage gefunden":
        answer = matched_question.get('answer', 'Keine Antwort verfügbar')

        # Simulation der Fragebeantwortung (Gewichtung/Aktivierung)
        activation = simulate_question_answering(category_nodes, matched_question['question'], questions)

        # Rückgabe der relevanten Informationen
        return f"Frage: \"{query}\"", f"Antwort: \"{answer}\"", f"Gewichtung: {activation:.2f}"
    else:
        # Falls keine passende Frage gefunden wurde
        return f"Frage: \"{query}\"", "Antwort: \"Keine passende Frage gefunden\"", "Gewichtung: 0.00"

def gradio_interface(query):
    """
    Gradio-Schnittstelle zur Verarbeitung der Benutzerabfrage.

    Args:
        query (str): Die Abfrage des Benutzers.

    Returns:
        tuple: Die gefundene Frage, Antwort und Gewichtung.
    """
    if category_nodes and questions:
        return test_model_with_answers(category_nodes, questions, query)
    else:
        logging.critical("Kein Modell gefunden.")
        return "Fehler", "Kein Modell geladen.", "0.00"

# Erstelle das Gradio-Interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(label="Frage eingeben", placeholder="Stellen Sie eine Frage..."),
    outputs=[
        gr.Textbox(label="Frage"),
        gr.Textbox(label="Antwort"),
        gr.Textbox(label="Gewichtung")
    ],
    title="Frage-Antwort-Modell",
    description="Stellen Sie eine Frage, und das Modell wird versuchen, eine passende Antwort mit Gewichtung zu finden."
)

# Starte das Gradio-Interface
iface.launch()
