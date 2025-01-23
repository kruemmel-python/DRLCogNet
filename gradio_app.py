import logging
import gradio as gr
import time
import json

from main import (
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

def measure_response_time(func, *args, **kwargs):
    """
    Misst die Zeit, die eine Funktion benötigt, um ausgeführt zu werden, und gibt die Ergebnisse zusammen mit der Zeit zurück.

    Args:
        func (callable): Die auszuführende Funktion.
        *args: Positionsargumente für die Funktion.
        **kwargs: Schlüsselwortargumente für die Funktion.

    Returns:
        tuple: Die Ergebnisse der Funktion und die verstrichene Zeit in Millisekunden.
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000  # Umwandlung in Millisekunden
    return result, elapsed_time

def extract_questions_and_answers_from_json(input_json, output_txt):
    """
    Extrahiert Fragen und Antworten aus einer JSON-Datei und schreibt sie in eine Textdatei.

    Args:
        input_json (str): Der Pfad zur Eingabe-JSON-Datei.
        output_txt (str): Der Pfad zur Ausgabe-Textdatei.
    """
    try:
        with open(input_json, mode='r', encoding='utf-8') as jsonfile, open(output_txt, mode='w', encoding='utf-8') as txtfile:
            data = json.load(jsonfile)
            questions = data.get('questions', [])
            for question in questions:
                q = question.get('question', '')
                a = question.get('answer', '')
                if q and a:
                    txtfile.write(f'"question": "{q}",\n')
                    txtfile.write(f'"answer": "{a}"\n\n')
        print(f"Fragen und Antworten wurden erfolgreich in {output_txt} geschrieben.")
    except FileNotFoundError:
        print(f"Die Datei {input_json} wurde nicht gefunden.")
    except json.JSONDecodeError:
        print(f"Fehler beim Parsen der JSON-Datei {input_json}.")
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")

def load_questions_and_answers(file_path):
    """
    Lädt Fragen und Antworten aus einer Textdatei.

    Args:
        file_path (str): Der Pfad zur Textdatei.

    Returns:
        str: Der Inhalt der Textdatei.
    """
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "Datei nicht gefunden."
    except Exception as e:
        return f"Fehler beim Lesen der Datei: {e}"

# Gradio-Interface
def gradio_interface(query):
    """
    Gradio-Schnittstelle zur Verarbeitung der Benutzerabfrage.

    Args:
        query (str): Die Abfrage des Benutzers.

    Returns:
        tuple: Die gefundene Frage, Antwort, Gewichtung und die verstrichene Zeit in Millisekunden.
    """
    if category_nodes and questions:
        result, elapsed_time = measure_response_time(test_model_with_answers, category_nodes, questions, query)
        return *result, f"Reaktionszeit: {elapsed_time:.2f} ms"
    else:
        logging.critical("Kein Modell gefunden.")
        return "Fehler", "Kein Modell geladen.", "0.00", "Reaktionszeit: 0.00 ms"

# Pfade zu den Dateien
input_json = 'model_with_qa.json'
output_txt = 'questions_and_answers.txt'

# Extrahiere Fragen und Antworten aus der JSON-Datei und speichere sie in der Textdatei
extract_questions_and_answers_from_json(input_json, output_txt)

# Lade die Fragen und Antworten aus der Textdatei
questions_and_answers_content = load_questions_and_answers(output_txt)

# Erstelle das Gradio-Interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(label="Frage eingeben", placeholder="Stellen Sie eine Frage..."),
    outputs=[
        gr.Textbox(label="Frage"),
        gr.Textbox(label="Antwort"),
        gr.Textbox(label="Gewichtung"),
        gr.Textbox(label="Reaktionszeit")
    ],
    title="Frage-Antwort-Modell",
    description="Stellen Sie eine Frage, und das Modell wird versuchen, eine passende Antwort mit Gewichtung zu finden."
)

# Füge ein aufklappbares Ausgabefenster hinzu, das die Liste der Fragen und Antworten anzeigt
with gr.Blocks() as demo:
    gr.Markdown("## Frage-Antwort-Modell")
    with gr.Row():
        with gr.Column():
            iface.render()
        with gr.Column():
            gr.Markdown("### Fragen und Antworten")
            gr.Textbox(value=questions_and_answers_content, lines=20, label="Fragen und Antworten", interactive=False)

# Starte das Gradio-Interface
demo.launch()
